import React, { useState, useRef, useEffect, useCallback } from 'react';

import readXlsxFile from 'read-excel-file';
import {
    Map,
    Database,
    Flame,
    Wind,
    Scan,
    Zap,
    Bot,
    CheckCircle,
    AlertCircle,
    Upload,
    FileText,
    Download,
    Eye,
    Layers,
    Search,
    Save,
    Check,
    Moon,
    Sun,
    ChevronDown,
    ChevronUp,
    Lock,
    Unlock
} from 'lucide-react';

import FusionChart from "./components/FusionChart";

// --- Types ---

type ResearchMethod =
    | 'database'
    | 'osm'
    | 'heat'
    | 'object_detection'
    | 'classification'
    | 'agent';

type AppMode = 'single' | 'batch';

interface ApiKeys {
    mapbox: string;
    aws: string;
    gemini: string;
    google_maps: string;
}

interface SinglePredictionResult {
    lat: number;
    lon: number;
    buffer: number;
    summary: string;
    timestamp: string;

    osm_html?: string;
    osm_class_counts?: Record<string, number>;
    osm_features?: Array<{
        lat: number;
        lon: number;
        category: string;
        name: string;
        tags: Record<string, any>;
    }>;

    database_records?: any[];

    classification_results?: any[];
    detection_results?: any[];

    satellite_image?: string;
    heatmap_image?: string;
    detection_image?: string;

    agent_report?: string;
    final_prediction?: string;

    agent_confidence?: number;
    agent_review_summary?: string;
    agent_rationale?: string;
    agent_key_evidence?: string[];
    agent_reviewed_inputs?: string[];

    heat_radiation_image?: string;
    heat_radiation_result?: any;

    fusion_result?: {
        classes: string[];
        scores: Record<string, number>;
        modal_scores?: {
            image?: Record<string, number>;
            osm?: Record<string, number>;
            database?: Record<string, number>;
            agent?: Record<string, number>;
        };
    };

    isError?: boolean;
}


interface BatchRow {
    lat: number;
    lon: number;
    buffer: number;
    status: 'done' | 'processing' | 'error';
    result?: string;
}

interface BatchJobState {
    jobId: string | null;
    fileName: string | null;
    rows: BatchRow[];
    loading: boolean;
    processed: boolean;
    progress: number;
    total: number;
    error: string | null;
}

const BATCH_STATE_KEY = 'energy-node-batch-job-v1';

const EMPTY_BATCH_JOB: BatchJobState = {
    jobId: null,
    fileName: null,
    rows: [],
    loading: false,
    processed: false,
    progress: 0,
    total: 0,
    error: null,
};

// Map methods to required keys (if any)
const methodRequirements: Partial<Record<ResearchMethod, keyof ApiKeys>> = {
    database: 'aws',
    classification: 'mapbox',
    object_detection: 'mapbox',
    agent: 'gemini',
};

// --- Mock Data Helpers ---

const MOCK_MARKDOWN = `
### Research Agent Report
**Location Analysis**: The coordinates point to an urban interface area.

*   **Vegetation**: High density of coniferous trees observed in the northern quadrant.
*   **Infrastructure**: Major highway detected within 2km buffer.
*   **Risk Assessment**: Moderate fire risk due to dry vegetation index history.

Recommended actions:
1.  Monitor gas emissions.
2.  Update OSM road network for new access paths.
`;

const MOCK_OBJ_DET_DATA = [
    { label: 'Car', confidence: 0.98, count: 14 },
    { label: 'Truck', confidence: 0.92, count: 3 },
    { label: 'Building', confidence: 0.99, count: 5 },
];

// --- Sub-Components ---

// 1. Sidebar Component
const Sidebar = ({
                     keys,
                     setKeys,
                     methods,
                     toggleMethod,
                     isDarkMode,
                     toggleTheme,
                     onConfirmConfig,
                     onUnlockConfig,
                     isConfigConfirmed,
                     validationStatus,
                     setValidationStatus,
                     clfModel,
                     setClfModel,
                     detModel,
                     setDetModel,
                     fusionWeights,
                     setFusionWeights
                 }: {
    keys: ApiKeys;
    setKeys: React.Dispatch<React.SetStateAction<ApiKeys>>;
    methods: Record<ResearchMethod, boolean>;
    toggleMethod: (m: ResearchMethod) => void;
    isDarkMode: boolean;
    toggleTheme: () => void;
    onConfirmConfig: () => void;
    onUnlockConfig: () => void;
    isConfigConfirmed: boolean;
    validationStatus: Record<string, 'idle' | 'checking' | 'valid' | 'invalid'>;
    setValidationStatus: React.Dispatch<React.SetStateAction<Record<string, 'idle' | 'checking' | 'valid' | 'invalid'>>>;
    clfModel: string;
    setClfModel: (v: string) => void;
    detModel: string;
    setDetModel: (v: string) => void;
    fusionWeights: {
        image: number;
        osm: number;
        database: number;
        agent: number;
    };
    setFusionWeights: React.Dispatch<
        React.SetStateAction<{
            image: number;
            osm: number;
            database: number;
            agent: number;
        }>
    >;
}) => {
    const [isKeysExpanded, setIsKeysExpanded] = useState(true);

    const methodList: { id: ResearchMethod; label: string; icon: React.ReactNode }[] = [
        { id: 'database', label: 'AWS Database', icon: <Database size={16} /> },
        { id: 'osm', label: 'Open Street Map', icon: <Map size={16} /> },
        { id: 'heat', label: 'Heat Radiation', icon: <Flame size={16} /> },
        { id: 'classification', label: 'Img Classification', icon: <Layers size={16} /> },
        { id: 'object_detection', label: 'Object Detection', icon: <Scan size={16} /> },
        { id: 'agent', label: 'Research Agent', icon: <Bot size={16} /> },
    ];

    const handleCheckKey = async (keyName: keyof ApiKeys) => {
        setValidationStatus(prev => ({ ...prev, [keyName]: 'checking' }));

        try {
            const response = await fetch('/api/validate-key', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    service: keyName === "google_maps" ? "google_maps" : keyName,
                    key: keys[keyName]
                })
            });

            const data = await response.json();

            setValidationStatus(prev => ({
                ...prev,
                [keyName]: data.valid ? 'valid' : 'invalid'
            }));

            if (!data.valid) {
                alert(`Validation Failed: ${data.message}`);
            }
        } catch (error) {
            console.error(error);
            setValidationStatus(prev => ({ ...prev, [keyName]: 'invalid' }));
            alert("Backend connection failed during validation.");
        }
    };

    const canConfirm = () => {
        const selectedMethods = (Object.keys(methods) as ResearchMethod[]).filter(m => methods[m]);
        for (const m of selectedMethods) {
            const reqKey = methodRequirements[m];
            if (reqKey) {
                if (validationStatus[reqKey] !== 'valid') {
                    return false;
                }
            }
        }
        return selectedMethods.length > 0;
    };

    const sidebarBg = isDarkMode ? 'bg-slate-900 border-slate-700' : 'bg-white border-slate-200 shadow-sm';
    const textPrimary = isDarkMode ? 'text-slate-100' : 'text-slate-900';
    const textSecondary = isDarkMode ? 'text-slate-400' : 'text-slate-500';
    const sectionTitle = isDarkMode ? 'text-slate-500' : 'text-slate-400';
    const inputBg = 'bg-slate-50';
    const inputBorder = isDarkMode ? 'border-slate-600' : 'border-slate-300';
    const inputText = 'text-slate-900';
    const hoverItem = isDarkMode ? 'hover:bg-slate-800' : 'hover:bg-slate-100';

    return (
        <div className={`w-80 flex flex-col h-screen fixed left-0 top-0 border-r transition-colors duration-300 ${sidebarBg} z-10 overflow-y-auto`}>
            <div className={`p-6 border-b ${isDarkMode ? 'border-slate-800' : 'border-slate-100'} flex justify-between items-start`}>
                <div>
                    <h1 className={`text-xl font-bold flex items-center gap-2 ${textPrimary}`}>
                        <Zap className="text-blue-500" fill="currentColor" />
                        Energy Node Classifier
                    </h1>
                </div>
            </div>

            <div className="p-6 space-y-6 flex-1">
                {/* API Keys Section */}
                <div className="space-y-4">
                    <div
                        className="flex justify-between items-center cursor-pointer group select-none"
                        onClick={() => setIsKeysExpanded(!isKeysExpanded)}
                    >
                        <h2 className={`text-xs font-bold uppercase tracking-wider transition-colors ${sectionTitle} group-hover:${textPrimary}`}>API Configuration</h2>
                        <div className="flex items-center gap-2">
                            <button
                                onClick={(e) => { e.stopPropagation(); toggleTheme(); }}
                                className={`p-1.5 rounded-full transition-colors ${isDarkMode ? 'bg-slate-800 text-yellow-400 hover:bg-slate-700' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'}`}
                                title="Toggle Theme"
                            >
                                {isDarkMode ? <Sun size={14} /> : <Moon size={14} />}
                            </button>
                            <div className={`p-1 rounded transition-colors ${isDarkMode ? 'text-slate-500 group-hover:text-slate-300' : 'text-slate-400 group-hover:text-slate-600'}`}>
                                {isKeysExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                            </div>
                        </div>
                    </div>

                    {isKeysExpanded && (
                        <div className="space-y-4 mt-2 animate-in fade-in slide-in-from-top-1 duration-200">
                            {(Object.keys(keys) as Array<keyof ApiKeys>)
                                .filter((k) => k !== 'google_maps')
                                .map((k) => (
                                    <div key={k} className="space-y-1">
                                    <label className={`text-xs font-medium capitalize ${textSecondary}`}>{k === 'aws'
                                        ? 'AWS DB'
                                        : k === 'google_maps'
                                            ? 'Google Maps'
                                            : k} Key</label>
                                    <div className="flex gap-2 items-center">
                                        <div className="relative flex-1">
                                            <input
                                                type="password"
                                                value={keys[k]}
                                                disabled={isConfigConfirmed} // Locks input when confirmed
                                                onChange={(e) => {
                                                    setKeys(prev => ({ ...prev, [k]: e.target.value }));
                                                    setValidationStatus(prev => ({ ...prev, [k]: 'idle' }));
                                                }}
                                                className={`w-full ${inputBg} border ${
                                                    validationStatus[k] === 'valid' ? 'border-green-500' :
                                                        validationStatus[k] === 'invalid' ? 'border-red-500' :
                                                            inputBorder
                                                } rounded-l px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none ${inputText} placeholder-slate-400 pr-8 disabled:opacity-50`}
                                                placeholder={`Enter ${k} key...`}
                                            />
                                            <div className="absolute right-2 top-2.5 pointer-events-none">
                                                {validationStatus[k] === 'checking' && <div className="animate-spin h-4 w-4 border-2 border-blue-500 border-t-transparent rounded-full"></div>}
                                                {validationStatus[k] === 'valid' && <CheckCircle size={16} className="text-green-500" />}
                                                {validationStatus[k] === 'invalid' && <AlertCircle size={16} className="text-red-500" />}
                                            </div>
                                        </div>
                                        <button
                                            onClick={() => handleCheckKey(k)}
                                            disabled={!keys[k] || validationStatus[k] === 'checking' || isConfigConfirmed}
                                            className={`${isDarkMode ? 'bg-slate-700 border-slate-600 text-slate-200' : 'bg-slate-200 border-slate-300 text-slate-700'} hover:opacity-90 px-3 py-2 rounded-r border border-l-0 disabled:opacity-50 transition-colors`}
                                            title="Validate Key"
                                        >
                                            <Check size={16} />
                                        </button>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                {/* Research Methods Section */}
                <div className={`space-y-3 pt-4 border-t ${isDarkMode ? 'border-slate-800' : 'border-slate-200'}`}>
                    <h2 className={`text-xs font-bold uppercase tracking-wider ${sectionTitle}`}>Research Methods</h2>
                    <div className="space-y-2">
                        {methodList.map((m) => {
                            const reqKey = methodRequirements[m.id];
                            const missingKey = reqKey && validationStatus[reqKey] !== 'valid';

                            return (
                                <div key={m.id}>
                                    <label className={`flex items-center space-x-3 cursor-pointer group ${hoverItem} p-2 rounded transition-colors ${isConfigConfirmed ? 'opacity-50 pointer-events-none' : ''}`}>
                                        <input
                                            type="checkbox"
                                            checked={methods[m.id]}
                                            disabled={isConfigConfirmed}
                                            onChange={() => toggleMethod(m.id)}
                                            className="form-checkbox h-4 w-4 text-blue-500 rounded border-slate-400 bg-white focus:ring-blue-500 focus:ring-offset-slate-900"
                                        />
                                        <div className={`flex items-center gap-2 text-sm ${textSecondary} group-hover:${textPrimary}`}>
                                            {m.icon}
                                            <span>{m.label}</span>
                                        </div>
                                    </label>

                                    {/* Model Selectors */}
                                    {m.id === 'classification' && methods.classification && (
                                        <div className="ml-9 mb-2 animate-in fade-in slide-in-from-top-1 duration-200">
                                            <select
                                                value={clfModel}
                                                onChange={e => setClfModel(e.target.value)}
                                                disabled={isConfigConfirmed}
                                                className={`w-full text-xs p-1.5 rounded border ${isDarkMode ? 'bg-slate-800 border-slate-600 text-slate-300' : 'bg-white border-slate-300 text-slate-700'}`}
                                            >
                                                <option value="convnext_large">ConvNeXtV2 (Large)</option>
                                                <option value="effnet">EfficientNetV2 M</option>
                                                <option value="resnet">ResNet-50</option>
                                                <option value="swin">SwinV2 (Large)</option>
                                            </select>
                                        </div>
                                    )}

                                    {m.id === 'object_detection' && methods.object_detection && (
                                        <div className="ml-9 mb-2 animate-in fade-in slide-in-from-top-1 duration-200">
                                            <select
                                                value={detModel}
                                                onChange={e => setDetModel(e.target.value)}
                                                disabled={isConfigConfirmed}
                                                className={`w-full text-xs p-1.5 rounded border ${isDarkMode ? 'bg-slate-800 border-slate-600 text-slate-300' : 'bg-white border-slate-300 text-slate-700'}`}
                                            >
                                                <option value="yolo11">YOLO v11</option>
                                                <option value="yolo26">YOLO v26</option>
                                            </select>
                                        </div>
                                    )}

                                    {/* HINT for missing keys */}
                                    {methods[m.id] && missingKey && (
                                        <div className="ml-9 text-xs text-red-500 flex items-center gap-1 mb-1">
                                            <AlertCircle size={10}/>
                                            <span>
  Requires valid {reqKey === "gemini" ? "Gemini" : reqKey} key
</span>
                                        </div>
                                    )}
                                </div>
                            );
                        })}
                    </div>
                </div>

                {/* Fusion Weights */}
                <div className="space-y-2 pt-4 border-t">
                    <h2 className="text-xs font-bold uppercase tracking-wider">Fusion Weights</h2>

                    {["image", "osm", "database", "agent"].map((key) => (
                        <div key={key} className="flex items-center gap-2">
                            <label className="text-xs w-20 capitalize">{key}</label>
                            <input
                                type="range"
                                min="0"
                                max="1"
                                step="0.05"
                                value={fusionWeights[key]}
                                onChange={(e) =>
                                    setFusionWeights(prev => ({
                                        ...prev,
                                        [key]: parseFloat(e.target.value)
                                    }))
                                }
                                className="flex-1"
                            />
                            <span className="text-xs font-mono w-10">
        {fusionWeights[key].toFixed(2)}
      </span>
                        </div>
                    ))}
                </div>

                {/* CONFIRMATION / UNLOCK BUTTON */}
                <div className="pt-4 mt-auto">
                    <button
                        onClick={isConfigConfirmed ? onUnlockConfig : onConfirmConfig}
                        disabled={!isConfigConfirmed && !canConfirm()}
                        className={`w-full py-2.5 rounded text-sm font-bold transition-all shadow-md flex items-center justify-center gap-2 ${
                            isConfigConfirmed
                                ? 'bg-amber-600 hover:bg-amber-700 text-white' // Unlock Color
                                : canConfirm()
                                    ? 'bg-blue-600 hover:bg-blue-700 text-white' // Confirm Color
                                    : 'bg-slate-300 text-slate-500 cursor-not-allowed'
                        }`}
                    >
                        {isConfigConfirmed ? (
                            <>
                                <Unlock size={16} />
                                Unlock & Edit
                            </>
                        ) : (
                            <>
                                <Lock size={16} />
                                Confirm & Lock
                            </>
                        )}
                    </button>

                    {!isConfigConfirmed && !canConfirm() && (
                        <p className="text-[10px] text-center mt-2 text-slate-400">
                            Select methods and validate required keys to continue.
                        </p>
                    )}
                </div>
            </div>
        </div>
    );
};

// Helper to clean up class names
const formatClassName = (name: string) => {
    if (!name) return 'Unknown';
    // Remove underscores, capitalize words
    return name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
};

const SingleMode = ({
                        methods,
                        onPredict,
                        loading,
                        result,
                        isDarkMode,
                        isConfigConfirmed
                    }: {
    methods: Record<ResearchMethod, boolean>;
    onPredict: (lat: number, lon: number, buf: number) => void;
    loading: boolean;
    result: SinglePredictionResult | null;
    isDarkMode: boolean;
    isConfigConfirmed: boolean;
}) => {
    const [lat, setLat] = useState<string>('');
    const [lon, setLon] = useState<string>('');
    // Default to 1.0 km
    const [buffer, setBuffer] = useState<string>('1.0');
    const [activeTab, setActiveTab] = useState<ResearchMethod | 'summary'>('summary');

    // GradCAM state: Hidden by default
    const [showExplanation, setShowExplanation] = useState(false);

    // --- OSM map filtering (iframe postMessage) ---
    const osmIframeRef = useRef<HTMLIFrameElement | null>(null);
    const [osmIframeReady, setOsmIframeReady] = useState(false);
    const [osmSelectedCats, setOsmSelectedCats] = useState<string[]>([]);

    // when we swap srcDoc, iframe reloads; mark as not ready until onLoad fires
    useEffect(() => {
        setOsmIframeReady(false);
    }, [result?.osm_html]);

    const postOsmFilter = (cats: string[]) => {
        const iframe = osmIframeRef.current;
        if (!iframe?.contentWindow) return;
        iframe.contentWindow.postMessage({ type: "OSM_FILTER", categories: cats }, "*");
    };

    // send whenever filter changes AND iframe is ready
    useEffect(() => {
        if (activeTab !== "osm") return;
        if (!osmIframeReady) return;
        postOsmFilter(osmSelectedCats);
    }, [osmSelectedCats, activeTab, osmIframeReady]);

    // also send once right after iframe becomes ready (covers first paint)
    useEffect(() => {
        if (activeTab !== "osm") return;
        if (!osmIframeReady) return;
        postOsmFilter(osmSelectedCats);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [osmIframeReady, activeTab]);

    const toggleOsmCat = (cat: string) => {
        setOsmSelectedCats(prev => prev.includes(cat) ? prev.filter(x => x !== cat) : [...prev, cat]);
    };

    const clearOsmFilter = () => setOsmSelectedCats([]); // empty => show all

    const cardBg = isDarkMode ? 'bg-slate-900 border-slate-700' : 'bg-white border-slate-200';
    const textPrimary = isDarkMode ? 'text-slate-100' : 'text-slate-800';
    const textLabel = isDarkMode ? 'text-slate-400' : 'text-slate-600';
    const inputClass = `w-full border rounded px-3 py-2 focus:ring-2 focus:ring-blue-500 outline-none transition-colors ${
        isDarkMode
            ? 'bg-slate-800 border-slate-600 text-white placeholder-slate-500'
            : 'bg-white border-slate-300 text-slate-900 placeholder-slate-400'
    }`;

    // Buffer Options
    const bufferOptions = [0.8, 1., 1.2, 1.4, 1.5, 1.6, 0.3];

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (isConfigConfirmed) {
            onPredict(parseFloat(lat), parseFloat(lon), parseFloat(buffer));
        }
    };

    const getAvailableTabs = () => {
        const tabs = (Object.keys(methods) as ResearchMethod[]).filter(m => methods[m]);
        return tabs;
    };

    const parseAgentReport = (report?: string) => {
        const raw = (report || '').trim();
        if (!raw) {
            return { rationale: '', evidence: [] as string[] };
        }

        // Expected backend format:
        // Final: ... (confidence=...)
        //
        // <rationale>
        //
        // Evidence:
        // - ...
        const parts = raw.split(/\n\s*\nEvidence:\n/i);
        const headerAndRationale = parts[0] || '';
        const evidenceBlock = parts[1] || '';

        // Remove the first line "Final: ..."
        const lines = headerAndRationale.split('\n');
        const withoutFinal = lines.slice(1).join('\n').trim();

        const lower = withoutFinal.toLowerCase();
        const rationale =
            !withoutFinal || lower === 'none' || lower === 'null'
                ? ''
                : withoutFinal;

        const evidence = evidenceBlock
            .split('\n')
            .map(s => s.trim())
            .filter(s => s.startsWith('-'))
            .map(s => s.replace(/^-+\s*/, '').trim())
            .filter(Boolean);

        return { rationale, evidence };
    };

    // --- AGGREGATION LOGIC FOR OBJECT DETECTION ---
    const getDetectionStats = () => {
        // Safety check: if no results array, return empty
        if (!result || !result.detection_results || !Array.isArray(result.detection_results)) {
            return [];
        }

        const counts: Record<string, { count: number; sumConf: number }> = {};

        result.detection_results.forEach((d: any) => {
            // Get label or default to 'Unknown'
            const rawLabel = d.label || 'Unknown';
            const label = formatClassName(rawLabel);

            if (!counts[label]) {
                counts[label] = { count: 0, sumConf: 0 };
            }

            counts[label].count += 1;
            counts[label].sumConf += d.confidence;
        });

        // Convert dictionary to sorted array
        return Object.keys(counts)
            .map((label) => ({
                label,
                count: counts[label].count,
                avgConf: counts[label].sumConf / counts[label].count
            }))
            .sort((a, b) => b.count - a.count); // Sort desc by count
    };

    const detectionStats = getDetectionStats();

    return (
        <div className="space-y-6">
            <div className={`p-6 rounded-lg shadow-sm border ${cardBg}`}>
                <h2 className={`text-lg font-semibold mb-4 ${textPrimary}`}>Single Coordinate Prediction</h2>
                <form onSubmit={handleSubmit} className="flex flex-col md:flex-row gap-4 items-end">
                    <div className="flex-1 w-full">
                        <label className={`block text-sm font-medium ${textLabel} mb-1`}>Latitude</label>
                        <input
                            required type="number" step="any"
                            value={lat} onChange={(e) => setLat(e.target.value)}
                            className={`${inputClass} disabled:opacity-50 disabled:cursor-not-allowed`}
                            placeholder="e.g. 49.72935"
                            disabled={!isConfigConfirmed}
                        />
                    </div>
                    <div className="flex-1 w-full">
                        <label className={`block text-sm font-medium ${textLabel} mb-1`}>Longitude</label>
                        <input
                            required type="number" step="any"
                            value={lon} onChange={(e) => setLon(e.target.value)}
                            className={`${inputClass} disabled:opacity-50 disabled:cursor-not-allowed`}
                            placeholder="e.g. 5.90331"
                            disabled={!isConfigConfirmed}
                        />
                    </div>
                    <div className="flex-1 w-full">
                        <label className={`block text-sm font-medium ${textLabel} mb-1`}>Buffer (km)</label>
                        <select
                            value={buffer}
                            onChange={(e) => setBuffer(e.target.value)}
                            className={`${inputClass} disabled:opacity-50 disabled:cursor-not-allowed`}
                            disabled={!isConfigConfirmed}
                        >
                            {bufferOptions.map((opt) => (
                                <option key={opt} value={opt}>{opt} km ({Math.round(opt * 1000)}m)</option>
                            ))}
                        </select>
                    </div>

                    <div className="flex flex-col items-end">
                        <button
                            type="submit"
                            disabled={loading || !isConfigConfirmed}
                            className={`px-6 py-2 rounded font-medium flex items-center gap-2 shadow-sm transition-colors ${
                                loading || !isConfigConfirmed
                                    ? 'bg-slate-300 text-slate-500 cursor-not-allowed'
                                    : 'bg-blue-600 text-white hover:bg-blue-700'
                            }`}
                            title={!isConfigConfirmed ? "Please confirm configuration in sidebar first" : "Run prediction"}
                        >
                            {loading ? (
                                <span className="animate-spin">⏳</span>
                            ) : !isConfigConfirmed ? (
                                <Lock size={18} />
                            ) : (
                                <Search size={18} />
                            )}
                            Predict
                        </button>
                    </div>
                </form>

                {!isConfigConfirmed && (
                    <div className={`mt-4 p-3 rounded text-sm flex items-center justify-center gap-2 ${isDarkMode ? 'bg-amber-900/20 text-amber-200 border border-amber-800' : 'bg-amber-50 text-amber-800 border border-amber-200'}`}>
                        <AlertCircle size={16} />
                        <span>Please select your research methods and click <b>"Confirm & Lock"</b> in the sidebar to enable prediction.</span>
                    </div>
                )}
            </div>

            {result && (
                <div className={`rounded-lg shadow-sm border overflow-hidden min-h-[600px] flex flex-col ${cardBg}`}>
                    <div className={`flex border-b overflow-x-auto ${isDarkMode ? 'border-slate-700' : 'border-slate-200'}`}>
                        <button
                            onClick={() => setActiveTab('summary')}
                            className={`px-4 py-3 text-sm font-medium whitespace-nowrap border-b-2 transition-colors ${
                                activeTab === 'summary'
                                    ? 'border-blue-500 text-blue-500'
                                    : `border-transparent ${isDarkMode ? 'text-slate-400 hover:text-slate-200' : 'text-slate-500 hover:text-slate-700'}`
                            }`}
                        >
                            Summary
                        </button>
                        {getAvailableTabs().map(tab => (
                            <button
                                key={tab}
                                onClick={() => setActiveTab(tab)}
                                className={`px-4 py-3 text-sm font-medium whitespace-nowrap border-b-2 capitalize transition-colors ${
                                    activeTab === tab
                                        ? 'border-blue-500 text-blue-500'
                                        : `border-transparent ${isDarkMode ? 'text-slate-400 hover:text-slate-200' : 'text-slate-500 hover:text-slate-700'}`
                                }`}
                            >
                                {tab === 'osm' ? 'OSM' : tab.replace('_', ' ')}
                            </button>
                        ))}
                    </div>

                    <div className={`p-6 flex-1 ${isDarkMode ? 'bg-slate-950/50' : 'bg-slate-50'}`}>
                        {activeTab === 'summary' && (
                            <div className="space-y-4">
                                <h3 className={`text-lg font-bold ${textPrimary}`}>Prediction Summary</h3>
                                {result.isError ? (
                                    <div className={`border-l-4 border-red-500 p-4 ${isDarkMode ? 'bg-red-900/20 text-red-200' : 'bg-red-50 text-red-700'}`}>
                                        <p className="font-bold flex items-center gap-2"><AlertCircle size={18}/> Error</p>
                                        <p>{result.summary}</p>
                                    </div>
                                ) : (
                                    <div className={`border-l-4 border-blue-500 p-4 ${isDarkMode ? 'bg-blue-900/20 text-slate-300' : 'bg-blue-50 text-slate-700'}`}>
                                        <p>{result.summary}</p>
                                    </div>
                                )}
                                <div className="grid grid-cols-3 gap-4 mt-4">
                                    {['Latitude', 'Longitude', 'Buffer'].map((label, i) => (
                                        <div key={label} className={`p-4 rounded shadow-sm border ${isDarkMode ? 'bg-slate-800 border-slate-700' : 'bg-white border-slate-200'}`}>
                                            <div className={`text-xs uppercase ${textLabel}`}>{label}</div>
                                            <div className={`font-mono text-lg ${textPrimary}`}>
                                                {i === 0 ? result.lat : i === 1 ? result.lon : `${result.buffer} km`}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                                {/* FUSION VISUALIZATION */}
                                {result.fusion_result && (
                                    <div className={`mt-6 p-4 rounded border ${
                                        isDarkMode
                                            ? 'bg-slate-900 border-slate-700'
                                            : 'bg-white border-slate-200'
                                    }`}>
                                        <h4 className={`text-sm font-semibold mb-3 ${
                                            isDarkMode ? 'text-slate-200' : 'text-slate-800'
                                        }`}>
                                            Fusion Score Breakdown
                                        </h4>

                                        <FusionChart fusion={result.fusion_result} />
                                    </div>
                                )}
                            </div>
                        )}

                        {/* =========================
                OSM TAB
               ========================= */}
                        {activeTab === 'osm' && (
                            <div className="h-full flex flex-col">
                                <h3 className={`text-lg font-bold mb-4 ${textPrimary}`}>Open Street Map View</h3>

                                <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                                    {/* Map (3/4 width) */}
                                    <div
                                        className={`lg:col-span-3 h-[800px] rounded-lg flex items-center justify-center border relative overflow-hidden ${
                                            isDarkMode ? 'bg-slate-800 border-slate-700' : 'bg-slate-200 border-slate-300'
                                        }`}
                                    >
                                        {result.osm_html ? (
                                            <iframe
                                                ref={osmIframeRef}
                                                title="OSM Visualization"
                                                srcDoc={result.osm_html}
                                                className="w-full h-full border-none"
                                                sandbox="allow-scripts allow-popups"
                                                onLoad={() => setOsmIframeReady(true)}
                                            />
                                        ) : (
                                            <div className="flex flex-col items-center justify-center p-8 text-center">
                                                <Map size={48} className="text-slate-400 mb-2" />
                                                <p className={textLabel}>Map data is not available.</p>
                                                <p className="text-xs text-slate-500">Ensure backend is running and OSM method was selected.</p>
                                            </div>
                                        )}
                                    </div>

                                    {/* Right Panel */}
                                    <div
                                        className={`lg:col-span-1 h-[800px] rounded-lg border overflow-hidden flex flex-col ${
                                            isDarkMode ? 'bg-slate-900 border-slate-700' : 'bg-white border-slate-200'
                                        }`}
                                    >
                                        {/* Class counts header + FILTER UI */}
                                        <div className={`p-4 border-b ${isDarkMode ? 'border-slate-800' : 'border-slate-100'}`}>
                                            <div className="flex items-center justify-between">
                                                <h4 className={`text-xs font-bold uppercase tracking-wider ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>
                                                    Detected OSM Classes
                                                </h4>
                                                <span className={`text-xs font-mono ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>
                          {result.osm_class_counts ? Object.values(result.osm_class_counts).reduce((a, b) => a + b, 0) : 0}
                        </span>
                                            </div>

                                            {/* Filter controls */}
                                            <div className="flex items-center justify-between gap-2 mt-3">
                                                <button
                                                    onClick={clearOsmFilter}
                                                    className={`text-xs px-2 py-1 rounded border ${
                                                        isDarkMode
                                                            ? 'border-slate-700 text-slate-200 hover:bg-slate-800'
                                                            : 'border-slate-200 text-slate-700 hover:bg-slate-50'
                                                    }`}
                                                    title="Show all categories on the map"
                                                >
                                                    Show all
                                                </button>

                                                <div className={`text-[11px] ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>
                                                    {osmSelectedCats.length > 0 ? `Filtering: ${osmSelectedCats.length}` : 'Filtering: none'}
                                                </div>
                                            </div>

                                            {result.osm_class_counts && Object.keys(result.osm_class_counts).length > 0 ? (
                                                <ul className="space-y-2 mt-3">
                                                    {Object.entries(result.osm_class_counts)
                                                        .sort((a, b) => b[1] - a[1])
                                                        .map(([cls, count]) => {
                                                            const checked = osmSelectedCats.includes(cls);

                                                            return (
                                                                <li
                                                                    key={cls}
                                                                    className={`flex items-center justify-between gap-3 border-b pb-2 ${
                                                                        isDarkMode ? 'border-slate-800' : 'border-slate-100'
                                                                    }`}
                                                                >
                                                                    <label className="flex items-center gap-2 cursor-pointer select-none">
                                                                        <input
                                                                            type="checkbox"
                                                                            checked={checked}
                                                                            onChange={() => toggleOsmCat(cls)}
                                                                            className="h-4 w-4"
                                                                            title="Toggle this category visibility on the map"
                                                                        />
                                                                        <span className={`text-sm font-medium ${isDarkMode ? 'text-slate-200' : 'text-slate-900'}`}>
                                      {formatClassName(cls)}
                                    </span>
                                                                    </label>

                                                                    <span className={`text-sm font-mono ${isDarkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                                    {count}
                                  </span>
                                                                </li>
                                                            );
                                                        })}
                                                </ul>
                                            ) : (
                                                <div className={`text-sm mt-3 ${isDarkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                                                    No OSM classes detected.
                                                </div>
                                            )}

                                            <div className={`mt-3 text-[11px] leading-relaxed ${isDarkMode ? 'text-slate-500' : 'text-slate-500'}`}>
                                                Counts are computed from categorization rules (derived classes), not raw tag keys.
                                                <br />
                                                Use the checkboxes to filter what the map displays.
                                            </div>
                                        </div>

                                        {/* Feature details */}
                                        <div className="p-4 overflow-auto flex-1">
                                            <h4 className={`text-xs font-bold uppercase tracking-wider mb-3 ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>
                                                Feature Details (Unfold)
                                            </h4>

                                            {Array.isArray(result.osm_features) && result.osm_features.length > 0 ? (
                                                <div className="space-y-3">
                                                    {result.osm_features
                                                        .slice()
                                                        .sort((a, b) => (a.category || '').localeCompare(b.category || ''))
                                                        .map((f, idx) => {
                                                            const title = `${formatClassName(f.category)} — ${f.name || 'Unnamed'}`;
                                                            const tags = f.tags || {};
                                                            const keys = Object.keys(tags).sort();

                                                            // quick “fuel/source” highlight if present
                                                            const fuelKeys = [
                                                                "plant:source",
                                                                "generator:source",
                                                                "generator:method",
                                                                "generator:type",
                                                                "substance",
                                                                "content",
                                                                "coal:type"
                                                            ];
                                                            const fuelRows = fuelKeys
                                                                .filter(k => tags[k] !== undefined && tags[k] !== null && String(tags[k]).trim() !== "")
                                                                .map(k => [k, String(tags[k])] as [string, string]);

                                                            return (
                                                                <details
                                                                    key={idx}
                                                                    className={`rounded border ${
                                                                        isDarkMode ? 'border-slate-700 bg-slate-900/30' : 'border-slate-200 bg-white'
                                                                    }`}
                                                                >
                                                                    <summary
                                                                        className={`cursor-pointer select-none px-3 py-2 text-sm font-medium ${
                                                                            isDarkMode ? 'text-slate-200' : 'text-slate-900'
                                                                        }`}
                                                                    >
                                                                        {title}
                                                                        <div className={`text-[11px] font-mono mt-1 ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>
                                                                            {f.lat.toFixed(5)}, {f.lon.toFixed(5)}
                                                                        </div>
                                                                    </summary>

                                                                    <div className="px-3 pb-3 pt-1 space-y-3">
                                                                        {fuelRows.length > 0 && (
                                                                            <div
                                                                                className={`p-2 rounded border ${
                                                                                    isDarkMode ? 'border-slate-700 bg-slate-950/40' : 'border-slate-200 bg-slate-50'
                                                                                }`}
                                                                            >
                                                                                <div
                                                                                    className={`text-[11px] uppercase tracking-wider mb-2 ${
                                                                                        isDarkMode ? 'text-slate-400' : 'text-slate-500'
                                                                                    }`}
                                                                                >
                                                                                    Fuel / Source (important)
                                                                                </div>
                                                                                <div className="space-y-1">
                                                                                    {fuelRows.map(([k, v]) => (
                                                                                        <div key={k} className="flex justify-between gap-3 text-xs">
                                                                                            <span className={`font-mono ${isDarkMode ? 'text-slate-300' : 'text-slate-700'}`}>{k}</span>
                                                                                            <span className={`${isDarkMode ? 'text-slate-200' : 'text-slate-900'}`}>{v}</span>
                                                                                        </div>
                                                                                    ))}
                                                                                </div>
                                                                            </div>
                                                                        )}

                                                                        <div className={`text-[11px] uppercase tracking-wider ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>
                                                                            All Tags
                                                                        </div>

                                                                        <div className={`max-h-64 overflow-auto rounded border ${isDarkMode ? 'border-slate-700' : 'border-slate-200'}`}>
                                                                            <table className="w-full text-xs">
                                                                                <tbody>
                                                                                {keys.length > 0 ? (
                                                                                    keys.map((k) => (
                                                                                        <tr key={k} className={`${isDarkMode ? 'border-slate-800' : 'border-slate-100'} border-b`}>
                                                                                            <td
                                                                                                className={`px-2 py-1 font-mono align-top ${
                                                                                                    isDarkMode ? 'text-slate-300' : 'text-slate-700'
                                                                                                }`}
                                                                                            >
                                                                                                {k}
                                                                                            </td>
                                                                                            <td className={`px-2 py-1 align-top ${isDarkMode ? 'text-slate-200' : 'text-slate-900'}`}>
                                                                                                {String(tags[k])}
                                                                                            </td>
                                                                                        </tr>
                                                                                    ))
                                                                                ) : (
                                                                                    <tr>
                                                                                        <td className="px-2 py-2 text-slate-500">No tags available.</td>
                                                                                    </tr>
                                                                                )}
                                                                                </tbody>
                                                                            </table>
                                                                        </div>
                                                                    </div>
                                                                </details>
                                                            );
                                                        })}
                                                </div>
                                            ) : (
                                                <div className={`text-sm ${isDarkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                                                    No OSM features returned. Enable OSM and run prediction again.
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}

                        {activeTab === 'database' && (
                            <div className="h-full">
                                <h3 className={`text-lg font-bold mb-4 ${textPrimary}`}>Database Records (AWS)</h3>
                                <div className={`overflow-x-auto rounded border ${isDarkMode ? 'bg-slate-900 border-slate-700' : 'bg-white border-slate-200'}`}>
                                    <table className={`min-w-full divide-y ${isDarkMode ? 'divide-slate-700' : 'divide-slate-200'}`}>
                                        <thead className={isDarkMode ? 'bg-slate-800' : 'bg-slate-50'}>
                                        <tr>
                                            {['ID', 'Type', 'Name', 'Distance', 'Source', 'Time', 'Remarks'].map(h => (
                                                <th key={h} className={`px-6 py-3 text-left text-xs font-medium uppercase tracking-wider ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>{h}</th>
                                            ))}
                                        </tr>
                                        </thead>
                                        <tbody className={`${isDarkMode ? 'bg-slate-900 divide-slate-800' : 'bg-white divide-slate-200'} divide-y`}>
                                        {result.database_records && result.database_records.length > 0 ? (
                                            result.database_records.map((row) => (
                                                <tr key={row.id}>
                                                    <td className={`px-6 py-4 whitespace-nowrap text-sm ${isDarkMode ? 'text-slate-300' : 'text-slate-900'}`}>#{row.id}</td>
                                                    <td className={`px-6 py-4 whitespace-nowrap text-sm ${isDarkMode ? 'text-slate-400' : 'text-slate-600'}`}>{row.type}</td>
                                                    <td className={`px-6 py-4 whitespace-nowrap text-sm font-medium ${isDarkMode ? 'text-slate-200' : 'text-slate-900'}`}>{row.name}</td>
                                                    <td className={`px-6 py-4 whitespace-nowrap text-sm ${isDarkMode ? 'text-slate-400' : 'text-slate-600'}`}>{row.dist}</td>
                                                    <td className={`px-6 py-4 whitespace-nowrap text-sm ${isDarkMode ? 'text-slate-400' : 'text-slate-600'}`}>{row.source}</td>
                                                    <td className={`px-6 py-4 whitespace-nowrap text-sm ${isDarkMode ? 'text-slate-400' : 'text-slate-600'}`}>{row.ts}</td>
                                                    <td className={`px-6 py-4 whitespace-nowrap text-sm ${isDarkMode ? 'text-slate-400' : 'text-slate-600'}`}>{row.remarks}</td>
                                                </tr>
                                            ))
                                        ) : (
                                            <tr>
                                                <td colSpan={7} className="px-6 py-8 text-center text-slate-500">
                                                    No records found.
                                                </td>
                                            </tr>
                                        )}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        )}

                        {activeTab === 'heat' && (
                            <div className="h-full">
                                <h3 className={`text-lg font-bold mb-4 ${textPrimary}`}>Thermal Heat Map (Landsat)</h3>

                                {result.heat_radiation_result?.error && (
                                    <div className={`p-4 rounded border text-sm ${
                                        isDarkMode ? 'bg-red-900/20 border-red-800 text-red-200' : 'bg-red-50 border-red-200 text-red-700'
                                    }`}>
                                        <div className="font-semibold mb-1">Heat pipeline failed</div>
                                        <div className="font-mono text-xs whitespace-pre-wrap">{String(result.heat_radiation_result.error)}</div>
                                    </div>
                                )}

                                {!result.heat_radiation_result?.error && !result.heat_radiation_image && (
                                    <div className={`p-6 rounded-lg border text-sm ${
                                        isDarkMode ? 'bg-slate-900 border-slate-700 text-slate-300' : 'bg-white border-slate-200 text-slate-700'
                                    }`}>
                                        <p className="font-medium mb-1">No heat output</p>
                                        <p className={`${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>
                                            Enable “Heat Radiation” in the sidebar and run prediction again.
                                        </p>
                                    </div>
                                )}

                                {result.heat_radiation_image && (
                                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                                        <div className={`rounded-lg border overflow-hidden ${
                                            isDarkMode ? 'bg-slate-900 border-slate-700' : 'bg-white border-slate-200'
                                        }`}>
                                            <img
                                                src={`data:image/png;base64,${result.heat_radiation_image}`}
                                                alt="Landsat thermal heatmap"
                                                className="w-full h-full object-contain"
                                            />
                                        </div>

                                        <div className={`p-4 rounded border text-sm ${
                                            isDarkMode ? 'bg-slate-800 border-slate-700 text-slate-200' : 'bg-white border-slate-200 text-slate-700'
                                        }`}>
                                            <div className="font-semibold mb-3">Heat classification</div>

                                            {(() => {
                                                const ranked = result.heat_radiation_result?.heat_ranked?.[0];
                                                const topLabel = ranked?.[0];
                                                const topProb = ranked?.[1];
                                                return (
                                                    <div className="mb-4">
                                                        <div className="text-xs uppercase opacity-70">Top class</div>
                                                        <div className="font-mono text-base">
                                                            {topLabel ? String(topLabel) : "—"}{typeof topProb === "number" ? ` (${topProb.toFixed(2)})` : ""}
                                                        </div>
                                                    </div>
                                                );
                                            })()}

                                            <div className="font-semibold mb-2">Key metrics</div>
                                            {(() => {
                                                const f = result.heat_radiation_result?.thermal_features;
                                                const meta = result.heat_radiation_result?.heatmap_meta;

                                                const rows: Array<[string, any]> = [
                                                    ["Scenes used", f?.n_scenes_used],
                                                    ["ΔT95 median (°C)", f?.deltaT95_median],
                                                    ["ΔT99 max (°C)", f?.deltaT99_max],
                                                    ["Hot frac median", f?.hot_frac_median],
                                                    ["Hot frac max", f?.hot_frac_max],
                                                    ["Hot frac persist", f?.hot_frac_persist],
                                                    ["Cloud cover (%)", meta?.cloud_cover],
                                                    ["Scene date", meta?.datetime ? String(meta.datetime).slice(0, 10) : undefined],
                                                    ["Thermal asset", meta?.thermal_asset_used],
                                                ];

                                                return (
                                                    <div className="space-y-2">
                                                        {rows
                                                            .filter(([, v]) => v !== undefined && v !== null && v !== "")
                                                            .map(([k, v]) => (
                                                                <div key={k} className="flex justify-between gap-4">
                                                                    <div className="text-xs opacity-70">{k}</div>
                                                                    <div className="font-mono text-xs">
                                                                        {typeof v === "number" ? v.toFixed(3) : String(v)}
                                                                    </div>
                                                                </div>
                                                            ))}
                                                    </div>
                                                );
                                            })()}

                                            <details className="mt-4">
                                                <summary className="cursor-pointer text-xs uppercase tracking-wider opacity-80">
                                                    Raw heat JSON
                                                </summary>
                                                <pre className="mt-2 whitespace-pre-wrap font-mono text-xs leading-relaxed">
{JSON.stringify(result.heat_radiation_result, null, 2)}
                        </pre>
                                            </details>
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}

                        {activeTab === 'object_detection' && (
                            <div className="h-full">
                                <h3 className={`text-lg font-bold mb-4 ${textPrimary}`}>Object Detection Results (YOLO)</h3>
                                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                                    <div className={`rounded-lg border flex items-center justify-center overflow-hidden relative ${isDarkMode ? 'bg-slate-800 border-slate-700' : 'bg-slate-100 border-slate-300'}`}>
                                        {result.detection_image ? (
                                            <img
                                                src={`data:image/png;base64,${result.detection_image}`}
                                                alt="Detection"
                                                className="w-full h-full object-contain"
                                            />
                                        ) : (
                                            <div className="flex flex-col items-center justify-center p-8 text-center text-slate-500">
                                                <Scan size={48} className="mb-2" />
                                                <p>No detection image generated.</p>
                                            </div>
                                        )}
                                    </div>
                                    <div>
                                        <table className={`w-full text-sm text-left border rounded ${isDarkMode ? 'border-slate-700 bg-slate-800 text-slate-400' : 'border-slate-200 bg-white text-slate-500'}`}>
                                            <thead className={`text-xs uppercase border-b ${isDarkMode ? 'bg-slate-900 text-slate-400 border-slate-700' : 'bg-slate-50 text-slate-700'}`}>
                                            <tr>
                                                <th className="px-4 py-3">Object Type</th>
                                                <th className="px-4 py-3">Count</th>
                                                <th className="px-4 py-3">Avg. Conf</th>
                                            </tr>
                                            </thead>
                                            <tbody>
                                            {detectionStats.length > 0 ? (
                                                detectionStats.map((stat, i) => (
                                                    <tr key={i} className={`border-b ${isDarkMode ? 'hover:bg-slate-700 border-slate-700' : 'hover:bg-slate-50'}`}>
                                                        <td className={`px-4 py-3 font-medium ${isDarkMode ? 'text-slate-200' : 'text-slate-900'}`}>{stat.label}</td>
                                                        <td className="px-4 py-3 font-mono">{stat.count}</td>
                                                        <td className="px-4 py-3">{(stat.avgConf * 100).toFixed(1)}%</td>
                                                    </tr>
                                                ))
                                            ) : (
                                                <tr>
                                                    <td colSpan={3} className="px-4 py-8 text-center text-slate-500">
                                                        No objects detected above threshold.
                                                        {/* Debug helper: Only show if raw results exist but aggregation failed */}
                                                        {result.detection_results && result.detection_results.length > 0 && (
                                                            <div className="text-xs text-red-400 mt-2">
                                                                Raw detections exist but display failed.
                                                            </div>
                                                        )}
                                                    </td>
                                                </tr>
                                            )}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            </div>
                        )}

                        {activeTab === 'classification' && (
                            <div className="h-full flex flex-col">
                                <div className="flex justify-between items-center mb-4 gap-3">
                                    <h3 className={`text-lg font-bold ${textPrimary}`}>Scene Classification</h3>

                                    <div className="flex items-center gap-2">
                                        <button
                                            onClick={() => setShowExplanation(!showExplanation)}
                                            className={`flex items-center gap-2 px-3 py-1.5 rounded text-sm font-medium transition-colors border ${
                                                showExplanation
                                                    ? 'bg-purple-600 border-purple-600 text-white shadow-sm'
                                                    : `${isDarkMode ? 'bg-slate-800 border-slate-600 text-slate-300 hover:bg-slate-700' : 'bg-white border-slate-300 text-slate-700 hover:bg-slate-50'}`
                                            }`}
                                        >
                                            <Eye size={16}/>
                                            {showExplanation ? 'Hide GradCAM' : 'Show GradCAM'}
                                        </button>
                                    </div>
                                </div>

                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    {/* Satellite Image Container */}
                                    <div
                                        className={`h-[800px] rounded-lg border overflow-hidden relative flex items-center justify-center shadow-inner ${
                                            isDarkMode ? 'bg-slate-900 border-slate-700' : 'bg-slate-100 border-slate-200'
                                        }`}
                                    >
                                    {result.satellite_image ? (
                                            <>
                                                {/* Base Layer: Original Satellite Image */}
                                                <img
                                                    src={`data:image/png;base64,${result.satellite_image}`}
                                                    alt="Satellite Source"
                                                    className="absolute inset-0 w-full h-full object-contain z-10"
                                                />

                                                {/* Overlay Layer: GradCAM Heatmap */}
                                                {result.heatmap_image ? (
                                                    <img
                                                        src={`data:image/png;base64,${result.heatmap_image}`}
                                                        alt="GradCAM Heatmap"
                                                        className="absolute inset-0 w-full h-full object-contain z-20 mix-blend-screen transition-opacity duration-300 ease-in-out"
                                                        style={{ opacity: showExplanation ? 0.7 : 0 }}
                                                    />
                                                ) : null}

                                                {/* Fallback if toggle is on but no heatmap exists */}
                                                {!result.heatmap_image && showExplanation && (
                                                    <div
                                                        className="absolute z-30 bottom-2 right-2 px-2 py-1 bg-black/70 text-white text-xs rounded backdrop-blur-sm">
                                                        No Heatmap Available
                                                    </div>
                                                )}
                                            </>
                                        ) : (
                                            <div
                                                className="flex flex-col items-center justify-center p-8 text-center text-slate-500">
                                                <Layers size={48} className="mb-2 opacity-50"/>
                                                <p>Image data not available</p>
                                            </div>
                                        )}
                                    </div>

                                    <div
                                        className={`p-4 rounded border ${isDarkMode ? 'bg-slate-800 border-slate-700' : 'bg-white border-slate-200'}`}>
                                        <h4 className={`font-semibold mb-4 ${textPrimary}`}>Prediction
                                            Probabilities</h4>
                                        <div className="space-y-4">
                                            {(result?.classification_results && result.classification_results.length > 0
                                                    ? result.classification_results
                                                    : [{label: "No Data", score: 0}]
                                            ).map((item: any, idx: number) => (
                                                <div key={idx}>
                                                    <div className="flex justify-between text-sm mb-1">
                                                        <span
                                                            className={`font-medium ${isDarkMode ? 'text-slate-200' : 'text-slate-900'}`}>{formatClassName(item.label)}</span>
                                                        <span
                                                            className={isDarkMode ? 'text-slate-400' : 'text-slate-500'}>{(item.score * 100).toFixed(1)}%</span>
                                                    </div>
                                                    <div
                                                        className={`w-full rounded-full h-2 ${isDarkMode ? 'bg-slate-700' : 'bg-slate-200'}`}>
                                                        <div
                                                            className={`h-2 rounded-full ${idx === 0 ? 'bg-blue-600' : 'bg-slate-400'}`}
                                                            style={{width: `${item.score * 100}%`}}
                                                        ></div>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* ✅ Agent tab now renders backend output */}
                        {activeTab === 'agent' && (
                            <div className="h-full">
                                <h3 className={`text-lg font-bold mb-4 flex items-center gap-2 ${textPrimary}`}>
                                    <Bot className="text-blue-500"/>
                                    Research Agent Output
                                </h3>

                                {result.agent_report ? (
                                    <div className={`p-6 rounded-lg border text-sm ${
                                        isDarkMode ? 'bg-slate-900 border-slate-700 text-slate-200' : 'bg-white border-slate-200 text-slate-700'
                                    }`}>
                                        {/* Final prediction header */}
                                        {result.final_prediction && (
                                            <div className={`mb-4 p-3 rounded border ${
                                                isDarkMode ? 'bg-slate-800 border-slate-700' : 'bg-slate-50 border-slate-200'
                                            }`}>
                                                <div
                                                    className={`text-xs uppercase ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>
                                                    Final prediction
                                                </div>
                                                <div className={`text-lg font-semibold ${textPrimary}`}>
                                                    {formatClassName(result.final_prediction)}
                                                </div>

                                                {typeof result.agent_confidence === 'number' && (
                                                    <div className={`text-xs mt-1 ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>
                                                        Confidence: {(result.agent_confidence * 100).toFixed(1)}%
                                                    </div>
                                                )}
                                            </div>
                                        )}

                                        {/* Review summary */}
                                        <div className="space-y-5">
                                            <div>
                                                <div className={`text-xs uppercase mb-1 ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>
                                                    Review summary
                                                </div>
                                                <div className={`${isDarkMode ? 'text-slate-200' : 'text-slate-700'}`}>
                                                    {result.agent_review_summary?.trim()
                                                        ? result.agent_review_summary
                                                        : "No review summary returned by the agent."}
                                                </div>
                                            </div>

                                            {/* Rationale */}
                                            <div>
                                                <div className={`text-xs uppercase mb-1 ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>
                                                    Rationale
                                                </div>
                                                <div className={`${isDarkMode ? 'text-slate-200' : 'text-slate-700'}`}>
                                                    {result.agent_rationale?.trim()
                                                        ? result.agent_rationale
                                                        : "No rationale returned by the agent."}
                                                </div>
                                            </div>

                                            {/* Evidence list */}
                                            <div>
                                                <div className={`text-xs uppercase mb-1 ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>
                                                    Key evidence
                                                </div>
                                                {Array.isArray(result.agent_key_evidence) && result.agent_key_evidence.length > 0 ? (
                                                    <ul className={`list-disc pl-5 space-y-1 ${isDarkMode ? 'text-slate-200' : 'text-slate-700'}`}>
                                                        {result.agent_key_evidence.map((e, idx) => (
                                                            <li key={idx}>{e}</li>
                                                        ))}
                                                    </ul>
                                                ) : (
                                                    <div className={`${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>
                                                        No evidence list returned by the agent.
                                                    </div>
                                                )}
                                            </div>

                                            {/* Optional: show what it claims it reviewed */}
                                            {Array.isArray(result.agent_reviewed_inputs) && result.agent_reviewed_inputs.length > 0 && (
                                                <div>
                                                    <div className={`text-xs uppercase mb-1 ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>
                                                        Reviewed inputs
                                                    </div>
                                                    <div className={`${isDarkMode ? 'text-slate-300' : 'text-slate-600'}`}>
                                                        {result.agent_reviewed_inputs.join(', ')}
                                                    </div>
                                                </div>
                                            )}

                                            {/* Raw output collapsible */}
                                            <details className={`${isDarkMode ? 'text-slate-300' : 'text-slate-600'}`}>
                                                <summary className="cursor-pointer text-xs uppercase tracking-wider">
                                                    Raw agent output
                                                </summary>
                                                <pre className="mt-2 whitespace-pre-wrap font-mono text-xs leading-relaxed">
{result.agent_report}
                        </pre>
                                            </details>
                                        </div>
                                    </div>
                                ) : (
                                    <div className={`p-6 rounded-lg border text-sm ${
                                        isDarkMode ? 'bg-slate-900 border-slate-700 text-slate-300' : 'bg-white border-slate-200 text-slate-700'
                                    }`}>
                                        <p className="font-medium mb-1">No agent output</p>
                                        <p className={`${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>
                                            Enable “Research Agent”, validate your Gemini key, and run prediction again.
                                        </p>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};

// Batch Mode View
const BatchMode = ({
                       isDarkMode,
                       isConfigConfirmed,
                       batchJob,
                       setBatchJob,
                       onStartBatch,
                       onClearBatchJob
                   }: {
    isDarkMode: boolean;
    isConfigConfirmed: boolean;
    batchJob: BatchJobState;
    setBatchJob: React.Dispatch<React.SetStateAction<BatchJobState>>;
    onStartBatch: (file: File) => Promise<void>;
    onClearBatchJob: () => void;
}) => {
    const [file, setFile] = useState<File | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const cardBg = isDarkMode ? 'bg-slate-900 border-slate-700' : 'bg-white border-slate-200';
    const textPrimary = isDarkMode ? 'text-slate-100' : 'text-slate-800';

    const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        const uploadedFile = e.target.files?.[0];
        if (!uploadedFile) return;

        setFile(uploadedFile);

        if (uploadedFile.name.endsWith('.csv')) {
            const reader = new FileReader();
            reader.onload = (event) => {
                const text = event.target?.result as string;
                const lines = text.split('\n');
                const parsed: BatchRow[] = [];

                lines.forEach((line, idx) => {
                    if (!line.trim()) return;

                    // skip header row if present
                    if (idx === 0 && line.toLowerCase().includes('latitude')) return;

                    const [lat, lon, buff] = line.split(',').map(s => parseFloat(s.trim()));
                    if (!isNaN(lat) && !isNaN(lon) && !isNaN(buff)) {
                        if (buff <= 50) {
                            parsed.push({ lat, lon, buffer: buff, status: 'done' });
                        }
                    }
                });

                setBatchJob(prev => ({
                    ...prev,
                    rows: parsed,
                    fileName: uploadedFile.name,
                    progress: 0,
                    total: parsed.length,
                    error: null,
                    processed: false,
                    loading: false,
                    jobId: null,
                }));
            };
            reader.readAsText(uploadedFile);
        } else if (uploadedFile.name.match(/\.xlsx?$/)) {
            readXlsxFile(uploadedFile).then((sheetRows: any[]) => {
                const parsed: BatchRow[] = [];

                sheetRows.forEach((row, idx) => {
                    if (idx === 0 && String(row[0] ?? '').toLowerCase().includes('latitude')) {
                        return;
                    }

                    if (row.length >= 3) {
                        const lat = parseFloat(String(row[0]));
                        const lon = parseFloat(String(row[1]));
                        const buff = parseFloat(String(row[2]));
                        if (!isNaN(lat) && !isNaN(lon) && !isNaN(buff)) {
                            if (buff <= 50) {
                                parsed.push({ lat, lon, buffer: buff, status: 'done' });
                            }
                        }
                    }
                });

                setBatchJob(prev => ({
                    ...prev,
                    rows: parsed,
                    fileName: uploadedFile.name,
                    progress: 0,
                    total: parsed.length,
                    error: null,
                    processed: false,
                    loading: false,
                    jobId: null,
                }));
            }).catch((err: any) => {
                console.error("Error reading excel file:", err);
                alert("Error reading Excel file.");
            });
        }
    };

    const handleRunBatch = async () => {
        if (!file) return;
        await onStartBatch(file);
    };

    const downloadCSV = () => {
        if (!batchJob.jobId) return;
        window.open(`/api/batch/${batchJob.jobId}/download`, "_blank");
    };

    return (
        <div className="space-y-6">
            <div className={`p-6 rounded-lg shadow-sm border ${cardBg}`}>
                <h2 className={`text-lg font-semibold mb-4 ${textPrimary}`}>Batch Prediction</h2>

                <div className={`mb-6 p-4 rounded border text-sm ${
                    isDarkMode
                        ? 'bg-slate-800 border-slate-700 text-slate-300'
                        : 'bg-slate-50 border-slate-200 text-slate-700'
                }`}>
                    <div className="font-semibold mb-2">Input File Requirements</div>

                    <ul className="list-disc pl-5 space-y-1">
                        <li>Supported formats: <b>.csv</b> or <b>.xlsx</b></li>
                        <li>First row should contain headers</li>
                        <li>Required columns (case-sensitive):</li>
                    </ul>

                    <div className="mt-2 font-mono text-xs bg-black/80 text-white p-3 rounded">
                        {`latitude,longitude,buffer
52.535223392234045,13.243375821692387,1.4
53.165035,9.49738,1.0`}
                    </div>

                    <div className="mt-3 text-xs opacity-80">
                        • Decimal separator must be a dot (.)<br/>
                        • Buffer is in kilometers<br/>
                        • Choose between 0.3, 1.0, 1.2, 1.4 and 1.6<br/>
                        • No empty rows
                    </div>
                </div>

                {!isConfigConfirmed ? (
                    <div className="text-center py-20 text-slate-500 flex flex-col items-center">
                        <Lock className="mb-4 text-slate-400" size={48} />
                        <p className="font-medium">Configuration Locked</p>
                        <p className="text-sm mt-1">
                            Please confirm your API settings and research methods in the sidebar to enable batch processing.
                        </p>
                    </div>
                ) : (
                    <>
                        {!batchJob.loading && !batchJob.processed && (
                            <div className="space-y-4">
                                <div
                                    onClick={() => fileInputRef.current?.click()}
                                    className={`border-2 border-dashed rounded-lg p-10 flex flex-col items-center justify-center cursor-pointer transition-colors ${
                                        isDarkMode ? 'border-slate-700 hover:bg-slate-800' : 'border-slate-300 hover:bg-slate-50'
                                    }`}
                                >
                                    <Upload className="text-slate-400 mb-2" size={32} />
                                    <span className={`font-medium ${isDarkMode ? 'text-slate-300' : 'text-slate-600'}`}>
                                        Click to upload CSV or Excel
                                    </span>
                                    <span className="text-slate-400 text-sm mt-1">
                                        Format: lat, lon, buffer (Max 50km buffer)
                                    </span>
                                    <input
                                        type="file"
                                        ref={fileInputRef}
                                        accept=".csv, .xlsx, .xls"
                                        onChange={handleFileUpload}
                                        className="hidden"
                                    />
                                </div>

                                {batchJob.fileName && (
                                    <div className={`flex items-center justify-between p-3 rounded ${isDarkMode ? 'bg-slate-800' : 'bg-slate-100'}`}>
                                        <div className="flex items-center gap-2">
                                            <FileText size={18} className="text-blue-500" />
                                            <span className={`text-sm font-medium ${textPrimary}`}>{batchJob.fileName}</span>
                                            <span className="text-xs text-slate-500">({batchJob.rows.length} valid rows)</span>
                                        </div>
                                        <button
                                            onClick={handleRunBatch}
                                            disabled={batchJob.rows.length === 0 || batchJob.loading}
                                            className="bg-blue-600 text-white px-4 py-1.5 rounded text-sm hover:bg-blue-700 disabled:opacity-50"
                                        >
                                            Run Batch
                                        </button>
                                    </div>
                                )}
                            </div>
                        )}

                        {batchJob.loading && (
                            <div className="flex flex-col items-center justify-center py-20 text-center">
                                <div className="w-16 h-16 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin mb-4"></div>
                                <h3 className={`text-lg font-medium ${textPrimary}`}>Processing Batch Data...</h3>
                                <p className="text-slate-500 text-sm">
                                    {batchJob.total > 0
                                        ? `Progress: ${batchJob.progress}/${batchJob.total}`
                                        : 'Running background python tasks...'}
                                </p>
                            </div>
                        )}

                        {batchJob.error && !batchJob.loading && (
                            <div className={`mt-4 p-3 rounded text-sm ${
                                isDarkMode ? 'bg-red-900/20 text-red-200 border border-red-800' : 'bg-red-50 text-red-700 border border-red-200'
                            }`}>
                                {batchJob.error}
                            </div>
                        )}

                        {batchJob.processed && (
                            <div>
                                <div className="flex justify-between items-center mb-4">
                                    <div className="flex items-center gap-2 text-green-600">
                                        <CheckCircle size={20} />
                                        <span className="font-medium">Batch Processing Complete</span>
                                    </div>
                                    <button
                                        onClick={downloadCSV}
                                        className={`flex items-center gap-2 px-4 py-2 rounded text-sm transition-colors ${
                                            isDarkMode ? 'bg-slate-700 text-white hover:bg-slate-600' : 'bg-slate-800 text-white hover:bg-slate-900'
                                        }`}
                                    >
                                        <Download size={16} />
                                        Download CSV
                                    </button>
                                </div>

                                <div className={`max-h-[400px] overflow-auto border rounded ${isDarkMode ? 'border-slate-700' : 'border-slate-200'}`}>
                                    <table className={`w-full text-sm text-left ${isDarkMode ? 'text-slate-300' : 'text-slate-600'}`}>
                                        <thead className={`sticky top-0 ${isDarkMode ? 'bg-slate-800 text-slate-400' : 'bg-slate-50 text-slate-500'}`}>
                                        <tr>
                                            <th className="px-4 py-2">Lat</th>
                                            <th className="px-4 py-2">Lon</th>
                                            <th className="px-4 py-2">Buffer</th>
                                            <th className="px-4 py-2">Status</th>
                                            <th className="px-4 py-2">Result</th>
                                        </tr>
                                        </thead>
                                        <tbody>
                                        {batchJob.rows.map((r, i) => (
                                            <tr key={i} className={`border-b ${isDarkMode ? 'border-slate-800 hover:bg-slate-800/50' : 'border-slate-100 hover:bg-slate-50'}`}>
                                                <td className="px-4 py-2">{r.lat}</td>
                                                <td className="px-4 py-2">{r.lon}</td>
                                                <td className="px-4 py-2">{r.buffer}</td>
                                                <td className="px-4 py-2">
                                                        <span className="bg-green-100 text-green-800 text-xs px-2 py-0.5 rounded-full uppercase">
                                                            {r.status}
                                                        </span>
                                                </td>
                                                <td className="px-4 py-2">{r.result}</td>
                                            </tr>
                                        ))}
                                        </tbody>
                                    </table>
                                </div>

                                <button
                                    onClick={() => {
                                        setFile(null);
                                        onClearBatchJob();
                                    }}
                                    className="mt-4 text-sm text-blue-600 hover:underline"
                                >
                                    Start New Batch
                                </button>
                            </div>
                        )}
                    </>
                )}
            </div>
        </div>
    );
};

// Main App Layout
const App = () => {
    const [keys, setKeys] = useState<ApiKeys>({
        mapbox: '',
        aws: '',
        gemini: '',
        google_maps: ''
    });
    const [isDarkMode, setIsDarkMode] = useState(false);

    const [validationStatus, setValidationStatus] = useState<Record<string, 'idle' | 'checking' | 'valid' | 'invalid'>>({});
    const [isConfigConfirmed, setIsConfigConfirmed] = useState(false);

    const [methods, setMethods] = useState<Record<ResearchMethod, boolean>>({
        database: true,
        osm: false,
        heat: false,
        object_detection: false,
        classification: false,
        agent: false
    });

    const [fusionWeights, setFusionWeights] = useState({
        image: 0.4,
        osm: 0.25,
        database: 0.25,
        agent: 0.10
    });

    const [clfModel, setClfModel] = useState("convnext_large");
    const [detModel, setDetModel] = useState("yolo26");

    const [mode, setMode] = useState<AppMode>('single');
    const [singleLoading, setSingleLoading] = useState(false);
    const [singleResult, setSingleResult] = useState<SinglePredictionResult | null>(null);

    const [batchJob, setBatchJob] = useState<BatchJobState>(() => {
        if (typeof window === 'undefined') return EMPTY_BATCH_JOB;

        try {
            const raw = localStorage.getItem(BATCH_STATE_KEY);
            if (!raw) return EMPTY_BATCH_JOB;
            return { ...EMPTY_BATCH_JOB, ...JSON.parse(raw) };
        } catch {
            return EMPTY_BATCH_JOB;
        }
    });

    const batchPollRef = useRef<number | null>(null);

    const stopBatchPolling = useCallback(() => {
        if (batchPollRef.current !== null) {
            window.clearInterval(batchPollRef.current);
            batchPollRef.current = null;
        }
    }, []);

    const clearBatchJob = useCallback(() => {
        stopBatchPolling();
        setBatchJob(EMPTY_BATCH_JOB);
        if (typeof window !== 'undefined') {
            localStorage.removeItem(BATCH_STATE_KEY);
        }
    }, [stopBatchPolling]);

    useEffect(() => {
        if (typeof window !== 'undefined') {
            localStorage.setItem(BATCH_STATE_KEY, JSON.stringify(batchJob));
        }
    }, [batchJob]);

    const pollBatchJob = useCallback((id: string) => {
        stopBatchPolling();

        const tick = async () => {
            try {
                const res = await fetch(`/api/batch/${id}`);
                const data = await res.json();

                if (data?.error === 'not found') {
                    setBatchJob(prev => ({
                        ...prev,
                        loading: false,
                        error: 'Batch job not found on backend.',
                    }));
                    stopBatchPolling();
                    return;
                }

                setBatchJob(prev => ({
                    ...prev,
                    jobId: id,
                    loading: data.status === 'running',
                    processed: data.status === 'finished',
                    progress: typeof data.progress === 'number' ? data.progress : prev.progress,
                    total: typeof data.total === 'number' ? data.total : prev.total,
                    error: data.status === 'error' ? (data.error || 'Batch failed') : null,
                }));

                if (data.status === 'finished' || data.status === 'error') {
                    stopBatchPolling();
                }
            } catch (err) {
                console.error('Batch polling failed:', err);
            }
        };

        tick();
        batchPollRef.current = window.setInterval(tick, 2000);
    }, [stopBatchPolling]);

    useEffect(() => {
        return () => stopBatchPolling();
    }, [stopBatchPolling]);

    useEffect(() => {
        if (batchJob.jobId && !batchJob.processed && !batchJob.error && batchPollRef.current === null) {
            pollBatchJob(batchJob.jobId);
        }
    }, [batchJob.jobId, batchJob.processed, batchJob.error, pollBatchJob]);

    const startBatch = async (file: File) => {
        setBatchJob(prev => ({
            ...prev,
            loading: true,
            processed: false,
            progress: 0,
            error: null,
            total: prev.rows.length,
        }));

        try {
            const formData = new FormData();
            formData.append("file", file);

            formData.append("aws_api_key", keys.aws);
            formData.append("mapbox_api_key", keys.mapbox);
            formData.append("gemini_api_key", keys.gemini);
            formData.append("google_maps_api_key", keys.google_maps);

            formData.append("run_osm", String(methods.osm));
            formData.append("run_database", String(methods.database));
            formData.append("run_classification", String(methods.classification));
            formData.append("run_object_detection", String(methods.object_detection));
            formData.append("run_agent", String(methods.agent));
            formData.append("run_heat_radiation", String(methods.heat));

            formData.append("classification_model", clfModel);
            formData.append("detection_model", detModel);
            formData.append("fusion_weights", JSON.stringify(fusionWeights));

            const response = await fetch("/api/batch", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Failed to start batch: ${response.status}`);
            }

            const data = await response.json();
            const id = data.job_id;

            setBatchJob(prev => ({
                ...prev,
                jobId: id,
                loading: true,
                processed: false,
                progress: 0,
                error: null,
            }));

            pollBatchJob(id);
        } catch (err) {
            console.error(err);
            setBatchJob(prev => ({
                ...prev,
                loading: false,
                error: 'Failed to start batch process.',
            }));
        }
    };

    const toggleMethod = (m: ResearchMethod) => {
        setMethods(prev => ({ ...prev, [m]: !prev[m] }));
    };

    const toggleTheme = () => {
        setIsDarkMode(!isDarkMode);
    };

    const handleSinglePredict = async (lat: number, lon: number, buffer: number) => {
        setSingleLoading(true);
        setSingleResult(null);

        try {
            const apiUrl = `/api/predict`;

            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    latitude: lat,
                    longitude: lon,
                    buffer: buffer,

                    aws_api_key: keys.aws,
                    mapbox_api_key: keys.mapbox,
                    gemini_api_key: keys.gemini,
                    google_maps_api_key: keys.google_maps,

                    run_osm: methods.osm,
                    run_database: methods.database,
                    run_classification: methods.classification,
                    run_object_detection: methods.object_detection,
                    run_agent: methods.agent,
                    run_heat_radiation: methods.heat,

                    classification_model: clfModel,
                    detection_model: detModel,
                    fusion_weights: fusionWeights,
                }),
            });

            if (!response.ok) throw new Error(`HTTP Error: ${response.status}`);
            const data = await response.json();

            setSingleResult({
                lat, lon, buffer,
                timestamp: new Date().toISOString(),
                summary: data.summary || 'Analysis Complete',

                osm_html: data.osm_html,
                osm_class_counts: data.osm_class_counts,
                osm_features: data.osm_features,

                database_records: data.database_records,

                classification_results: data.classification_results,
                detection_results: data.detection_results,

                satellite_image: data.satellite_image,
                heatmap_image: data.heatmap_image,
                detection_image: data.detection_image,

                agent_report: data.agent_report,
                final_prediction: data.final_prediction,

                agent_confidence: data.agent_confidence,
                agent_review_summary: data.agent_review_summary,
                agent_rationale: data.agent_rationale,
                agent_key_evidence: data.agent_key_evidence,
                agent_reviewed_inputs: data.agent_reviewed_inputs,

                heat_radiation_image: data.heat_radiation_image,
                heat_radiation_result: data.heat_radiation_result,
                fusion_result: data.fusion_result,
            });
        } catch (error) {
            console.error('Prediction Error:', error);
            setSingleResult({
                lat, lon, buffer,
                timestamp: new Date().toISOString(),
                summary: "Error connecting to backend.",
                isError: true
            });
        } finally {
            setSingleLoading(false);
        }
    };

    return (
        <div className={`flex min-h-screen font-sans transition-colors duration-200 ${isDarkMode ? 'bg-slate-950 text-slate-100' : 'bg-slate-50 text-slate-900'}`}>
            <Sidebar
                keys={keys}
                setKeys={setKeys}
                methods={methods}
                toggleMethod={toggleMethod}
                isDarkMode={isDarkMode}
                toggleTheme={toggleTheme}
                isConfigConfirmed={isConfigConfirmed}
                onConfirmConfig={() => setIsConfigConfirmed(true)}
                onUnlockConfig={() => setIsConfigConfirmed(false)}
                validationStatus={validationStatus}
                setValidationStatus={setValidationStatus}
                clfModel={clfModel}
                setClfModel={setClfModel}
                detModel={detModel}
                setDetModel={setDetModel}
                fusionWeights={fusionWeights}
                setFusionWeights={setFusionWeights}
            />

            <main className="ml-80 flex-1 p-8">
                <div className="flex justify-between items-center mb-8">
                    <div>
                        <h2 className={`text-2xl font-bold ${isDarkMode ? 'text-slate-100' : 'text-slate-800'}`}>Dashboard</h2>
                        <p className={`text-sm ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>
                            Configure analysis parameters and review predictions.
                        </p>
                    </div>

                    <div className={`p-1 rounded-lg flex ${isDarkMode ? 'bg-slate-800' : 'bg-slate-200'}`}>
                        <button
                            onClick={() => setMode('single')}
                            className={`px-4 py-2 text-sm font-medium rounded-md transition-all ${
                                mode === 'single'
                                    ? `${isDarkMode ? 'bg-slate-700 text-white shadow' : 'bg-white shadow text-slate-900'}`
                                    : `${isDarkMode ? 'text-slate-400 hover:text-slate-200' : 'text-slate-600 hover:text-slate-800'}`
                            }`}
                        >
                            Single Prediction
                        </button>
                        <button
                            onClick={() => setMode('batch')}
                            className={`px-4 py-2 text-sm font-medium rounded-md transition-all ${
                                mode === 'batch'
                                    ? `${isDarkMode ? 'bg-slate-700 text-white shadow' : 'bg-white shadow text-slate-900'}`
                                    : `${isDarkMode ? 'text-slate-400 hover:text-slate-200' : 'text-slate-600 hover:text-slate-800'}`
                            }`}
                        >
                            Batch Processing
                        </button>
                    </div>
                </div>

                {batchJob.loading && (
                    <div className={`mb-6 p-3 rounded border text-sm flex items-center justify-between ${
                        isDarkMode
                            ? 'bg-blue-900/20 border-blue-800 text-blue-200'
                            : 'bg-blue-50 border-blue-200 text-blue-800'
                    }`}>
                        <span>
                            Batch job running in background: {batchJob.progress}/{batchJob.total}
                        </span>
                        <button
                            onClick={() => setMode('batch')}
                            className="underline underline-offset-2"
                        >
                            Open batch view
                        </button>
                    </div>
                )}

                {mode === 'single' ? (
                    <SingleMode
                        methods={methods}
                        onPredict={handleSinglePredict}
                        loading={singleLoading}
                        result={singleResult}
                        isDarkMode={isDarkMode}
                        isConfigConfirmed={isConfigConfirmed}
                    />
                ) : (
                    <BatchMode
                        isDarkMode={isDarkMode}
                        isConfigConfirmed={isConfigConfirmed}
                        batchJob={batchJob}
                        setBatchJob={setBatchJob}
                        onStartBatch={startBatch}
                        onClearBatchJob={clearBatchJob}
                    />
                )}
            </main>
        </div>
    );
};


export default App;
