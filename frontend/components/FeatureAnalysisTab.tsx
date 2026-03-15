import React, { useState } from 'react';
import type { Feature } from '../types';
import { FeatureImportance } from '../types';

interface FeatureAnalysisTabProps {
  features: Feature[];
  satelliteImageUrl: string;
  gradCamImageUrl: string;
  lrpImageUrl: string;
}

type HeatmapMethod = 'gradCam' | 'lrp';

const getImportanceColor = (importance: FeatureImportance) => {
  switch (importance) {
    case FeatureImportance.High: return 'bg-red-500/10 text-red-700 dark:bg-red-500/20 dark:text-red-400';
    case FeatureImportance.Medium: return 'bg-yellow-500/10 text-yellow-700 dark:bg-yellow-500/20 dark:text-yellow-400';
    case FeatureImportance.Low: return 'bg-sky-500/10 text-sky-700 dark:bg-sky-500/20 dark:text-sky-400';
    default: return 'bg-slate-500/10 text-slate-700 dark:bg-slate-500/20 dark:text-slate-400';
  }
};

const ImageryAnalysisCard: React.FC<{ gradCamImageUrl: string; lrpImageUrl: string; satelliteImageUrl: string; }> = ({ gradCamImageUrl, lrpImageUrl, satelliteImageUrl }) => {
    const [heatmapMethod, setHeatmapMethod] = useState<HeatmapMethod>('gradCam');
    const [opacity, setOpacity] = useState(0.6);
    const activeHeatmapUrl = heatmapMethod === 'gradCam' ? gradCamImageUrl : lrpImageUrl;
    
    return (
        <div className="bg-slate-100 dark:bg-slate-800/50 rounded-lg border border-slate-200 dark:border-slate-700 h-full flex flex-col">
            <div className="p-3 border-b border-slate-200 dark:border-slate-700 flex-shrink-0">
                <div className="flex justify-between items-center">
                    <h3 className="text-sm font-semibold text-slate-800 dark:text-slate-200">
                        Visual Feature Importance
                    </h3>
                     <div className="inline-flex rounded-md shadow-sm bg-slate-200 dark:bg-slate-800 border border-slate-300 dark:border-slate-600 p-1">
                        <button onClick={() => setHeatmapMethod('gradCam')} className={`px-2 py-0.5 text-xs font-medium rounded ${heatmapMethod === 'gradCam' ? 'bg-blue-600 text-white' : 'text-slate-600 dark:text-slate-400 hover:bg-slate-300 dark:hover:bg-slate-700'}`}>
                            Grad-CAM
                        </button>
                        <button onClick={() => setHeatmapMethod('lrp')} className={`px-2 py-0.5 text-xs font-medium rounded ${heatmapMethod === 'lrp' ? 'bg-blue-600 text-white' : 'text-slate-600 dark:text-slate-400 hover:bg-slate-300 dark:hover:bg-slate-700'}`}>
                            LRP
                        </button>
                    </div>
                </div>
                <div className="mt-3 flex items-center space-x-3">
                    <label htmlFor="opacity-slider" className="text-xs font-medium text-slate-600 dark:text-slate-400">Heatmap Opacity</label>
                    <input
                        id="opacity-slider"
                        type="range"
                        min="0"
                        max="1"
                        step="0.05"
                        value={opacity}
                        onChange={(e) => setOpacity(parseFloat(e.target.value))}
                        className="w-full h-1.5 bg-slate-300 dark:bg-slate-600 rounded-full appearance-none cursor-pointer accent-blue-500 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-100 dark:focus:ring-offset-slate-800 focus:ring-blue-500"
                    />
                    <span className="text-xs font-mono text-slate-500 dark:text-slate-400 w-10 text-center">{Math.round(opacity * 100)}%</span>
                </div>
            </div>
            <div className="p-2 flex-grow relative aspect-square">
                <img 
                    src={satelliteImageUrl}
                    alt="Satellite view of the analysis area"
                    className="absolute inset-0 w-full h-full object-cover rounded-b-lg"
                />
                <img 
                    src={activeHeatmapUrl}
                    alt={`${heatmapMethod === 'gradCam' ? 'Grad-CAM' : 'LRP'} heatmap overlay`}
                    className="absolute inset-0 w-full h-full object-cover mix-blend-screen rounded-b-lg"
                    style={{ opacity }}
                />
            </div>
        </div>
    );
};


export const FeatureAnalysisTab: React.FC<FeatureAnalysisTabProps> = ({ features, satelliteImageUrl, gradCamImageUrl, lrpImageUrl }) => {
  return (
    <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
      <div className="bg-slate-100 dark:bg-slate-800/50 rounded-lg border border-slate-200 dark:border-slate-700 overflow-hidden xl:col-span-1">
        <table className="w-full text-sm text-left text-slate-600 dark:text-slate-300">
          <thead className="text-xs text-slate-500 dark:text-slate-400 uppercase bg-slate-200/60 dark:bg-slate-800">
            <tr>
              <th scope="col" className="px-6 py-3">Feature</th>
              <th scope="col" className="px-6 py-3">Value</th>
              <th scope="col" className="px-6 py-3">Importance</th>
            </tr>
          </thead>
          <tbody>
            {features.map((feature, index) => (
              <tr key={index} className="border-t border-slate-200 dark:border-slate-700 hover:bg-slate-200/50 dark:hover:bg-slate-800/60 transition-colors">
                <td className="px-6 py-4 font-medium text-slate-800 dark:text-slate-100">{feature.name}</td>
                <td className="px-6 py-4 font-mono">{feature.value}</td>
                <td className="px-6 py-4">
                  <span className={`px-2 py-1 text-xs font-semibold rounded-full ${getImportanceColor(feature.importance)}`}>
                    {feature.importance}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="xl:col-span-1">
        <ImageryAnalysisCard 
            satelliteImageUrl={satelliteImageUrl} 
            gradCamImageUrl={gradCamImageUrl}
            lrpImageUrl={lrpImageUrl} 
        />
      </div>
    </div>
  );
};