import React, { useState, useRef } from 'react';
import type { QueryHistoryItem, BatchJob } from '../types';
import { NodeType, BatchJobStatus } from '../types';
import { MapPinIcon, UploadIcon, FileTextIcon, CheckCircleIcon, DownloadIcon, PencilIcon, SunIcon, MoonIcon } from './icons';

interface LeftPanelProps {
  history: QueryHistoryItem[];
  onAnalyze: (coords: string, buffer: number) => void;
  onSelectHistoryItem: (id: string) => void;
  activeQueryId?: string | null;
  isLoading: boolean;
  batchJobs: BatchJob[];
  onBatchUpload: (file: File) => void;
  onDownloadResults: (jobId: string) => void;
  theme: 'light' | 'dark';
  onToggleTheme: () => void;
}

const getNodeTypeColor = (nodeType: NodeType) => {
  switch (nodeType) {
    case NodeType.Kraftwerk: return 'bg-red-500/10 text-red-600 dark:bg-red-500/20 dark:text-red-400 border border-red-500/20 dark:border-red-500/30';
    case NodeType.Industrie: return 'bg-purple-500/10 text-purple-600 dark:bg-purple-500/20 dark:text-purple-400 border border-purple-500/20 dark:border-purple-500/30';
    case NodeType.Verteilnetz: return 'bg-slate-500/10 text-slate-600 dark:bg-slate-500/20 dark:text-slate-400 border border-slate-500/20 dark:border-slate-500/30';
    case NodeType.Biogasanlage: return 'bg-green-500/10 text-green-600 dark:bg-green-500/20 dark:text-green-400 border border-green-500/20 dark:border-green-500/30';
    case NodeType.Speicher: return 'bg-cyan-500/10 text-cyan-600 dark:bg-cyan-500/20 dark:text-cyan-400 border border-cyan-500/20 dark:border-cyan-500/30';
    case NodeType.Verdichterstation: return 'bg-yellow-500/10 text-yellow-600 dark:bg-yellow-500/20 dark:text-yellow-400 border border-yellow-500/20 dark:border-yellow-500/30';
    default: return 'bg-gray-500/10 text-gray-600 dark:bg-gray-500/20 dark:text-gray-400 border border-gray-500/20 dark:border-gray-500/30';
  }
};

const HistoryItem: React.FC<{ item: QueryHistoryItem; isActive: boolean; onClick: () => void; }> = ({ item, isActive, onClick }) => {
    const formattedDate = new Intl.DateTimeFormat('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: 'numeric',
        minute: '2-digit',
        hour12: true,
    }).format(item.timestamp);

    const displayType = item.manualNodeType || item.nodeType;

    return (
        <li
            onClick={onClick}
            className={`p-3 rounded-lg cursor-pointer transition-colors duration-200 ${isActive ? 'bg-blue-500/10 dark:bg-blue-600/20 ring-1 ring-blue-500' : 'hover:bg-slate-200/50 dark:hover:bg-slate-700/50'}`}
        >
            <div className="flex justify-between items-center">
                <div>
                    <span className="font-mono text-sm font-medium text-slate-800 dark:text-slate-200">{item.coordinates}</span>
                    <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                        {formattedDate}
                        <span className="mx-2">|</span>
                        {item.bufferKm} km buffer
                    </p>
                </div>
                <div className={`flex items-center space-x-1.5 px-2 py-0.5 text-xs font-semibold rounded-full ${getNodeTypeColor(displayType)}`}>
                    {item.manualNodeType && <PencilIcon className="w-3 h-3" />}
                    <span>{displayType}</span>
                </div>
            </div>
        </li>
    );
};

const BatchJobItem: React.FC<{ job: BatchJob, onDownload: (jobId: string) => void }> = ({ job, onDownload }) => {
    const progressPercentage = job.total > 0 ? (job.progress / job.total) * 100 : 0;
    
    return (
        <div className="bg-slate-100 dark:bg-slate-800/50 p-3 rounded-lg border border-slate-200 dark:border-slate-700">
            <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3 min-w-0">
                    <FileTextIcon className="w-5 h-5 text-slate-500 dark:text-slate-400 flex-shrink-0" />
                    <p className="text-sm text-slate-700 dark:text-slate-300 truncate font-medium" title={job.fileName}>{job.fileName}</p>
                </div>
                {job.status === BatchJobStatus.Completed && (
                    <button onClick={() => onDownload(job.id)} className="p-1 rounded-md hover:bg-slate-300 dark:hover:bg-slate-600 transition-colors">
                        <DownloadIcon className="w-5 h-5 text-green-500 dark:text-green-400"/>
                    </button>
                )}
            </div>
            
            {job.status === BatchJobStatus.Processing && (
                <div className="mt-2">
                    <div className="flex justify-between text-xs text-slate-500 dark:text-slate-400 mb-1">
                        <span>{job.status}</span>
                        <span>{job.progress}/{job.total}</span>
                    </div>
                    <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-1.5">
                        <div className="bg-blue-500 h-1.5 rounded-full" style={{ width: `${progressPercentage}%` }}></div>
                    </div>
                </div>
            )}
             {job.status === BatchJobStatus.Completed && (
                <div className="mt-2 flex items-center space-x-2 text-xs text-green-600 dark:text-green-400">
                   <CheckCircleIcon className="w-4 h-4"/>
                   <span>Completed</span>
                </div>
            )}
            {job.status === BatchJobStatus.Error && (
                <div className="mt-2 text-xs text-red-500 dark:text-red-400">
                   <p><strong>Error:</strong> {job.error}</p>
                </div>
            )}
        </div>
    );
};


export const LeftPanel: React.FC<LeftPanelProps> = ({ history, onAnalyze, onSelectHistoryItem, activeQueryId, isLoading, batchJobs, onBatchUpload, onDownloadResults, theme, onToggleTheme }) => {
  const [coords, setCoords] = useState('');
  const [bufferRadius, setBufferRadius] = useState('1');
  const [coordError, setCoordError] = useState<string | null>(null);
  const [bufferError, setBufferError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateCoordinates = (value: string): string | null => {
    if (!value) {
        return 'Coordinates are required.';
    }
    const parts = value.split(',').map(p => p.trim());
    if (parts.length !== 2) {
        return 'Invalid format. Use "latitude, longitude".';
    }
    const lat = parseFloat(parts[0]);
    const lon = parseFloat(parts[1]);

    if (isNaN(lat) || isNaN(lon)) {
        return 'Coordinates must be numeric.';
    }
    
    // EU Bounding Box
    if (lat < 34 || lat > 72) {
        return 'Latitude is outside the valid range for Europe (34° to 72°).';
    }
    if (lon < -25 || lon > 45) {
        return 'Longitude is outside the valid range for Europe (-25° to 45°).';
    }

    return null; // No error
  };

  const validateBuffer = (value: string): string | null => {
    if (!value) {
        return 'Buffer radius is required.';
    }
    const num = parseFloat(value);
    if (isNaN(num)) {
        return 'Buffer must be a numeric value.';
    }
    if (num <= 0) {
        return 'Buffer must be greater than 0.';
    }
    if (num > 50) {
        return 'Buffer must be 50 km or less.';
    }
    return null;
  };


  const handleCoordsChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setCoords(value);
    if (value.trim() === '') {
        setCoordError(null);
    } else {
        setCoordError(validateCoordinates(value));
    }
  };

  const handleBufferChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setBufferRadius(value);
    if (value.trim() === '') {
        setBufferError(null);
    } else {
        setBufferError(validateBuffer(value));
    }
  };

  const handleAnalyzeClick = () => {
    const cError = validateCoordinates(coords);
    const bError = validateBuffer(bufferRadius);
    setCoordError(cError);
    setBufferError(bError);
      if (!isLoading && !cError && !bError) {
          onAnalyze(coords, parseFloat(bufferRadius));
      }
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setSelectedFile(event.target.files[0]);
    }
  };

  const handleUploadClick = () => {
    if (selectedFile) {
      onBatchUpload(selectedFile);
      setSelectedFile(null);
      if(fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  return (
    <aside className="w-[30%] max-w-[450px] h-full bg-slate-50 dark:bg-slate-800/50 border-r border-slate-200 dark:border-slate-700 p-6 flex flex-col space-y-6 overflow-y-auto">
      <header className="flex justify-between items-start">
        <div>
          <h1 className="text-2xl font-bold text-slate-800 dark:text-slate-100">Network Node Classifier</h1>
          <p className="text-slate-500 dark:text-slate-400">Analyze single nodes or process batches via CSV.</p>
        </div>
        <button onClick={onToggleTheme} className="p-2 rounded-full text-slate-500 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">
          {theme === 'dark' ? <SunIcon className="w-6 h-6 text-yellow-400" /> : <MoonIcon className="w-6 h-6 text-slate-700" />}
        </button>
      </header>

      {/* New Query Section */}
      <div className="space-y-4">
        <h2 className="text-lg font-semibold text-slate-700 dark:text-slate-200">Single Node Analysis</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="md:col-span-2">
                <label htmlFor="coords-input" className="block text-sm font-medium text-slate-600 dark:text-slate-400 mb-1">
                    Coordinates (Lat, Lon)
                </label>
                <input
                    id="coords-input"
                    type="text"
                    value={coords}
                    onChange={handleCoordsChange}
                    placeholder="e.g., 52.516, 13.377"
                    className={`w-full bg-white dark:bg-slate-900/50 border rounded-md px-3 py-2 text-slate-800 dark:text-slate-200 placeholder-slate-400 dark:placeholder-slate-500 focus:ring-2 focus:outline-none transition ${
                        coordError 
                        ? 'border-red-500 focus:ring-red-500/50 focus:border-red-500' 
                        : 'border-slate-300 dark:border-slate-600 focus:ring-blue-500 focus:border-blue-500'
                    }`}
                />
            </div>
            <div>
                 <label htmlFor="buffer-input" className="block text-sm font-medium text-slate-600 dark:text-slate-400 mb-1">
                    Buffer (km)
                </label>
                 <input
                    id="buffer-input"
                    type="number"
                    value={bufferRadius}
                    onChange={handleBufferChange}
                    placeholder="1"
                    className={`w-full bg-white dark:bg-slate-900/50 border rounded-md px-3 py-2 text-slate-800 dark:text-slate-200 placeholder-slate-400 dark:placeholder-slate-500 focus:ring-2 focus:outline-none transition ${
                        bufferError 
                        ? 'border-red-500 focus:ring-red-500/50 focus:border-red-500' 
                        : 'border-slate-300 dark:border-slate-600 focus:ring-blue-500 focus:border-blue-500'
                    }`}
                />
            </div>
        </div>
        {(coordError || bufferError) && (
            <div className="text-xs text-red-500 dark:text-red-400 -mt-2">
                {coordError && <p>{coordError}</p>}
                {bufferError && <p>{bufferError}</p>}
            </div>
        )}
        <button 
          onClick={handleAnalyzeClick}
          disabled={isLoading || !coords || !!coordError || !bufferRadius || !!bufferError}
          className="w-full bg-blue-600 text-white font-bold py-2.5 px-4 rounded-md hover:bg-blue-700 disabled:bg-slate-400 dark:disabled:bg-slate-600 disabled:cursor-not-allowed transition-colors flex items-center justify-center"
        >
          {isLoading ? (
            <>
              <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Analyzing...
            </>
          ) : 'Analyze Node'}
        </button>
      </div>

      {/* Batch Processing */}
      <div className="space-y-4">
        <h2 className="text-lg font-semibold text-slate-700 dark:text-slate-200">Batch Processing</h2>
        <div className="bg-slate-100 dark:bg-slate-900/50 p-4 rounded-lg border border-slate-200 dark:border-slate-700 space-y-3">
          <p className="text-xs text-slate-500 dark:text-slate-400">Upload a CSV file with headers: <code className="font-mono bg-slate-200 dark:bg-slate-700 p-1 rounded">id,latitude,longitude</code></p>
          <input type="file" accept=".csv" ref={fileInputRef} onChange={handleFileChange} className="hidden" id="csv-upload" />
          <div className="flex space-x-2">
            <label htmlFor="csv-upload" className="flex-grow text-center cursor-pointer bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-200 font-semibold py-2 px-4 rounded-md hover:bg-slate-300 dark:hover:bg-slate-600 transition-colors">
              {selectedFile ? selectedFile.name : 'Choose File'}
            </label>
            <button onClick={handleUploadClick} disabled={!selectedFile} className="bg-indigo-600 text-white font-bold py-2 px-4 rounded-md hover:bg-indigo-700 disabled:bg-slate-400 dark:disabled:bg-slate-600 disabled:cursor-not-allowed transition-colors flex items-center">
              <UploadIcon className="w-5 h-5 mr-2" />
              Process
            </button>
          </div>
        </div>
        <div className="space-y-3">
          {batchJobs.map(job => (
              <BatchJobItem key={job.id} job={job} onDownload={onDownloadResults}/>
          ))}
        </div>
      </div>
      
      {/* Query History Section */}
      <div className="flex-grow flex flex-col min-h-0">
        <h2 className="text-lg font-semibold text-slate-700 dark:text-slate-200 mb-4">Query History</h2>
        <div className="overflow-y-auto p-1 pr-2 -mr-2 flex-grow">
          {history.length > 0 ? (
            <ul className="space-y-3">
              {history.map(item => (
                <HistoryItem 
                  key={item.id} 
                  item={item} 
                  isActive={item.id === activeQueryId}
                  onClick={() => onSelectHistoryItem(item.id)}
                />
              ))}
            </ul>
          ) : (
            <div className="text-center text-slate-400 dark:text-slate-500 py-8">No single queries yet.</div>
          )}
        </div>
      </div>
    </aside>
  );
};