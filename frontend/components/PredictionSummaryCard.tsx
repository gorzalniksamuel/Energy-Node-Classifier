import React from 'react';
import { NodeType } from '../types';

interface PredictionSummaryCardProps {
  nodeType: NodeType;
  manualNodeType?: NodeType;
  confidence: number;
  onManualChange: (newType: NodeType) => void;
}

export const PredictionSummaryCard: React.FC<PredictionSummaryCardProps> = ({ nodeType, manualNodeType, confidence, onManualChange }) => {
  const confidenceColor = confidence > 90 ? 'bg-green-500' : confidence > 80 ? 'bg-yellow-500' : 'bg-orange-500';

  const displayType = manualNodeType || nodeType;

  const handleSelectChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const newType = event.target.value as NodeType;
    onManualChange(newType);
  };

  return (
    <div className="bg-slate-100 dark:bg-slate-800/50 rounded-lg p-6 border border-slate-200 dark:border-slate-700">
      <div className="grid md:grid-cols-2 gap-6">
        <div>
          <h3 className="text-sm font-medium text-slate-500 dark:text-slate-400 mb-1 flex items-center">
            {manualNodeType ? 'Manually Classified as' : 'Predicted Node Type'}
            {manualNodeType && <span className="ml-2 text-xs bg-yellow-500/10 text-yellow-700 dark:bg-yellow-500/20 dark:text-yellow-400 px-2 py-0.5 rounded-full">Overridden</span>}
          </h3>
          <div className="relative group">
            <select
              value={displayType}
              onChange={handleSelectChange}
              className="appearance-none w-full bg-transparent text-3xl font-bold text-green-600 dark:text-green-400 mt-1 pr-8 focus:outline-none cursor-pointer"
            >
              {Object.values(NodeType).map((type) => (
                <option key={type} value={type} className="bg-slate-100 dark:bg-slate-800 text-base font-medium">
                  {type}
                </option>
              ))}
            </select>
            <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-slate-400 dark:text-slate-400 group-hover:text-slate-600 dark:group-hover:text-slate-200 transition-colors">
                <svg className="fill-current h-6 w-6" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z"/></svg>
            </div>
          </div>
           {manualNodeType && (
            <p className="text-xs text-slate-400 dark:text-slate-500 mt-2">Original prediction: {nodeType}</p>
          )}
        </div>
        <div>
          <h3 className="text-sm font-medium text-slate-500 dark:text-slate-400">Confidence Score</h3>
          <div className="flex items-center mt-2">
            <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2.5 mr-4">
              <div
                className={`${confidenceColor} h-2.5 rounded-full`}
                style={{ width: `${confidence}%` }}
              ></div>
            </div>
            <span className="text-xl font-bold text-slate-800 dark:text-slate-200">{confidence}%</span>
          </div>
        </div>
      </div>
    </div>
  );
};