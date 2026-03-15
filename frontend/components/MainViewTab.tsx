import React, { useState } from 'react';
import { MapIcon, PhotographIcon } from './icons';

interface MainViewTabProps {}

type ImagerySource = 'google' | 'sentinel';

const ViewCard: React.FC<{ title: string; children: React.ReactNode; }> = ({ title, children }) => (
    <div className="bg-slate-100 dark:bg-slate-800/50 rounded-lg border border-slate-200 dark:border-slate-700 h-full flex flex-col">
        <h3 className="text-sm font-semibold text-slate-800 dark:text-slate-200 p-3 border-b border-slate-200 dark:border-slate-700 flex-shrink-0">{title}</h3>
        <div className="flex-grow p-2 relative">
            {children}
        </div>
    </div>
);

const MapPlaceholder = () => (
    <div className="absolute inset-0 bg-slate-200 dark:bg-slate-700/50 overflow-hidden rounded-b-lg">
        {/* Abstract Map SVG background */}
        <svg width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <pattern id="grid" width="30" height="30" patternUnits="userSpaceOnUse">
                    <path d="M 30 0 L 0 0 0 30" fill="none" stroke="rgba(100, 116, 139, 0.4)" strokeWidth="0.5"/>
                </pattern>
            </defs>
            <rect width="100%" height="100%" fill="url(#grid)" />
            <path d="M -20 80 C 80 120, 120 40, 220 80 S 320 220, 420 180" stroke="#4f46e5" strokeWidth="4" fill="none" opacity="0.5" />
            <path d="M 40 250 C 120 180, 160 280, 280 220 S 380 140, 450 180" stroke="#ca8a04" strokeWidth="3" fill="none" opacity="0.4" />
            <rect x="150" y="90" width="40" height="25" fill="rgba(71, 85, 105, 0.6)" />
            <rect x="230" y="160" width="60" height="35" fill="rgba(71, 85, 105, 0.6)" />
        </svg>
        {/* Center Marker and Buffer */}
        <div className="absolute inset-0 flex items-center justify-center">
            <div className="w-24 h-24 rounded-full bg-blue-500/20 animate-pulse"></div>
            <div className="absolute w-4 h-4 rounded-full bg-red-500 border-2 border-white shadow-lg"></div>
        </div>
    </div>
);

export const MainViewTab: React.FC<MainViewTabProps> = () => {
    const [imagerySource, setImagerySource] = useState<ImagerySource>('google');

    return (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-[500px]">
            <ViewCard title="Map View">
                <MapPlaceholder />
                <div className="absolute inset-0 bg-gradient-to-t from-black/5 to-transparent pointer-events-none rounded-b-lg"></div>
                <div className="absolute bottom-4 left-4 text-slate-500 dark:text-slate-400 text-xs font-mono">
                    <p>Interactive Map Placeholder</p>
                    <p>OSM Features</p>
                    <p>Analysis Buffer</p>
                </div>
            </ViewCard>
            <div className="flex flex-col h-full">
                <div className="flex-shrink-0 mb-2">
                    <div className="inline-flex rounded-md shadow-sm bg-slate-100 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700 p-1">
                        <button onClick={() => setImagerySource('google')} className={`px-3 py-1 text-sm font-medium rounded ${imagerySource === 'google' ? 'bg-blue-600 text-white' : 'text-slate-600 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-700'}`}>Google Satellite</button>
                        <button onClick={() => setImagerySource('sentinel')} className={`px-3 py-1 text-sm font-medium rounded ${imagerySource === 'sentinel' ? 'bg-blue-600 text-white' : 'text-slate-600 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-700'}`}>Sentinel-2 (True Color)</button>
                    </div>
                </div>
                <ViewCard title="Imagery View">
                     <div className="absolute inset-0 flex items-center justify-center text-slate-300 dark:text-slate-600">
                        <PhotographIcon className="w-32 h-32" />
                    </div>
                    <div className="absolute inset-0 bg-gradient-to-t from-black/5 to-transparent pointer-events-none rounded-b-lg"></div>
                </ViewCard>
            </div>
        </div>
    );
};