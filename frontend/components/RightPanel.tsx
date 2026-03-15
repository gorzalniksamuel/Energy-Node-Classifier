import React, { useState } from 'react';
import type { QueryResult, NodeType } from '../types';
import { DownloadIcon, MapPinIcon } from './icons';
import { PredictionSummaryCard } from './PredictionSummaryCard';
import { MainViewTab } from './MainViewTab';
import { FeatureAnalysisTab } from './FeatureAnalysisTab';
import { RawDataTab } from './RawDataTab';

interface RightPanelProps {
  result: QueryResult | null;
  isLoading: boolean;
  onManualOverride: (newType: NodeType) => void;
}

type Tab = 'main' | 'features' | 'raw';

const TabButton: React.FC<{ active: boolean; onClick: () => void; children: React.ReactNode }> = ({ active, onClick, children }) => (
    <button
        onClick={onClick}
        className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${active ? 'bg-blue-600 text-white' : 'text-slate-500 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-700'}`}
    >
        {children}
    </button>
);


export const RightPanel: React.FC<RightPanelProps> = ({ result, isLoading, onManualOverride }) => {
  const [activeTab, setActiveTab] = useState<Tab>('main');

  if (isLoading) {
    return (
      <main className="w-[70%] h-full p-8 flex items-center justify-center bg-white dark:bg-slate-900">
        <div className="text-center">
          <svg className="animate-spin mx-auto h-12 w-12 text-blue-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          <h2 className="mt-4 text-xl font-semibold text-slate-800 dark:text-slate-200">Analyzing node...</h2>
          <p className="text-slate-500 dark:text-slate-400">Fetching geospatial data and satellite imagery.</p>
        </div>
      </main>
    );
  }

  if (!result) {
    return (
      <main className="w-[70%] h-full p-8 flex items-center justify-center bg-white dark:bg-slate-900">
        <div className="text-center">
          <MapPinIcon className="mx-auto h-12 w-12 text-slate-300 dark:text-slate-600" />
          <h2 className="mt-4 text-xl font-semibold text-slate-500 dark:text-slate-400">Network Node Classifier</h2>
          <p className="text-slate-400 dark:text-slate-500">Enter coordinates to start analysis.</p>
        </div>
      </main>
    );
  }

  return (
    <main className="w-[70%] h-full p-8 flex flex-col bg-white dark:bg-slate-900 overflow-y-auto">
      {/* Top Bar */}
      <div className="flex justify-between items-center mb-6">
        <div className="flex items-center space-x-2">
            <MapPinIcon className="w-5 h-5 text-slate-500 dark:text-slate-400"/>
            <h2 className="text-xl font-bold font-mono text-slate-800 dark:text-slate-200">{result.coordinates}</h2>
        </div>
        <div className="relative group">
            <button className="flex items-center space-x-2 bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 text-slate-700 dark:text-slate-200 font-semibold py-2 px-4 rounded-md transition-colors">
                <DownloadIcon />
                <span>Export</span>
            </button>
            <div className="absolute right-0 mt-2 w-48 bg-slate-100 dark:bg-slate-700 rounded-md shadow-lg py-1 z-10 opacity-0 group-hover:opacity-100 transition-opacity duration-200 transform scale-95 group-hover:scale-100 origin-top-right">
                <a href="#" className="block px-4 py-2 text-sm text-slate-700 dark:text-slate-200 hover:bg-slate-200 dark:hover:bg-slate-600">Export as PDF</a>
                <a href="#" className="block px-4 py-2 text-sm text-slate-700 dark:text-slate-200 hover:bg-slate-200 dark:hover:bg-slate-600">Download GeoJSON</a>
            </div>
        </div>
      </div>

      {/* Prediction Summary */}
      <div className="mb-6">
        <PredictionSummaryCard 
            nodeType={result.nodeType} 
            manualNodeType={result.manualNodeType}
            confidence={result.confidence}
            onManualChange={onManualOverride}
        />
      </div>

      {/* Tabs */}
      <div className="flex space-x-2 mb-4">
        <TabButton active={activeTab === 'main'} onClick={() => setActiveTab('main')}>
          Main View
        </TabButton>

        <TabButton active={activeTab === 'features'} onClick={() => setActiveTab('features')}>
          Feature Analysis
        </TabButton>

        <TabButton active={activeTab === 'raw'} onClick={() => setActiveTab('raw')}>
          Raw Data
        </TabButton>
      </div>

      {/* Tab Content */}
      <div className="flex-grow">
        {activeTab === 'main' && <MainViewTab />}
        {activeTab === 'features' && <FeatureAnalysisTab features={result.features} satelliteImageUrl={result.satelliteImageUrl} gradCamImageUrl={result.gradCamImageUrl} lrpImageUrl={result.lrpImageUrl} />}
        {activeTab === 'raw' && <RawDataTab rawData={result.rawData} />}
      </div>
    </main>
  );
};