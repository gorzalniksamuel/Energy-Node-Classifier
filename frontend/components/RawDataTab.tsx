

import React from 'react';
import type { RawDataEntry } from '../types';
import { ChevronDownIcon } from './icons';

interface RawDataTabProps {
  rawData: RawDataEntry[];
}

const RawDataEntryItem: React.FC<{ entry: RawDataEntry }> = ({ entry }) => (
    <details className="bg-slate-100 dark:bg-slate-800/50 rounded-lg border border-slate-200 dark:border-slate-700 overflow-hidden">
        <summary className="flex justify-between items-center px-4 py-3 cursor-pointer hover:bg-slate-200/50 dark:hover:bg-slate-800 transition-colors">
            <span className="font-medium text-slate-800 dark:text-slate-200">{entry.source}</span>
            <ChevronDownIcon className="w-5 h-5 text-slate-500 dark:text-slate-400 transform transition-transform open:rotate-180" />
        </summary>
        <div className="p-4 border-t border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900/70">
            <pre className="text-xs text-slate-600 dark:text-slate-300 whitespace-pre-wrap font-mono overflow-x-auto">
                {JSON.stringify(entry.data, null, 2)}
            </pre>
        </div>
    </details>
);

export const RawDataTab: React.FC<RawDataTabProps> = ({ rawData }) => {
  return (
    <div className="space-y-4">
      {rawData.map((entry, index) => (
        <RawDataEntryItem key={index} entry={entry} />
      ))}
    </div>
  );
};