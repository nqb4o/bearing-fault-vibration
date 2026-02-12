import React, { useState } from 'react';
import { Play, DownloadCloud, Database, CheckCircle, RefreshCw, Layers } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { TrainingLog, TrainingConfig } from '../types';
import { startModelTraining, triggerDatasetDownload } from '../services/mockApiService';

const TrainingPanel: React.FC = () => {
  const [logs, setLogs] = useState<TrainingLog[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [dataStatus, setDataStatus] = useState<'idle' | 'downloading' | 'ready'>('idle');
  const [sampleCount, setSampleCount] = useState(0);
  
  // Config State
  const [epochs, setEpochs] = useState(15);
  const [batchSize, setBatchSize] = useState(32);

  const handleDownloadData = async () => {
    setDataStatus('downloading');
    try {
      const result = await triggerDatasetDownload();
      setSampleCount(result.samples);
      setDataStatus('ready');
    } catch (e) {
      console.error(e);
      setDataStatus('idle');
    }
  };

  const handleStartTraining = async () => {
    if (dataStatus !== 'ready') return;
    
    setIsTraining(true);
    setLogs([]); // Reset logs
    
    try {
      await startModelTraining({ epochs, batchSize }, (newLog) => {
        setLogs(prev => [...prev, newLog]);
      });
    } catch (e) {
      console.error(e);
    } finally {
      setIsTraining(false);
    }
  };

  return (
    <div className="grid grid-cols-12 gap-6">
      
      {/* --- Column 1: Configuration & Data --- */}
      <div className="col-span-12 lg:col-span-4 space-y-6">
        
        {/* Data Pipeline Card */}
        <div className="bg-white p-6 rounded-2xl border border-gray-100 shadow-sm">
          <div className="flex items-center mb-4 text-gray-800">
             <Database className="mr-2 text-google-blue" size={20} />
             <h2 className="font-semibold text-lg">Data Pipeline</h2>
          </div>
          <p className="text-sm text-gray-500 mb-6">
            Fetch standard vibration dataset from Kaggle, apply sliding window (1024), and fit StandardScaler.
          </p>

          <div className="space-y-4">
             <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <span className="text-sm font-medium text-gray-600">Source</span>
                <span className="text-xs bg-gray-200 text-gray-700 px-2 py-1 rounded">Kaggle/sumairaziz</span>
             </div>
             <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <span className="text-sm font-medium text-gray-600">Samples</span>
                <span className="text-sm font-bold text-gray-800">{sampleCount > 0 ? sampleCount.toLocaleString() : '-'}</span>
             </div>
          </div>

          <button 
            onClick={handleDownloadData}
            disabled={dataStatus === 'downloading' || isTraining}
            className={`mt-6 w-full flex items-center justify-center py-2.5 rounded-lg text-sm font-medium transition-all
              ${dataStatus === 'ready' 
                ? 'bg-green-50 text-green-700 border border-green-200 cursor-default' 
                : 'bg-google-blue text-white hover:bg-blue-700 active:scale-95'}
              disabled:opacity-50 disabled:cursor-not-allowed
            `}
          >
            {dataStatus === 'downloading' ? (
              <><RefreshCw className="animate-spin mr-2" size={16} /> Processing...</>
            ) : dataStatus === 'ready' ? (
              <><CheckCircle className="mr-2" size={16} /> Data Ready</>
            ) : (
              <><DownloadCloud className="mr-2" size={16} /> Download & Process</>
            )}
          </button>
        </div>

        {/* Hyperparameters Card */}
        <div className="bg-white p-6 rounded-2xl border border-gray-100 shadow-sm">
          <div className="flex items-center mb-4 text-gray-800">
             <Layers className="mr-2 text-google-blue" size={20} />
             <h2 className="font-semibold text-lg">Training Config</h2>
          </div>
          
          <div className="space-y-6">
            <div>
              <label className="flex justify-between text-sm font-medium text-gray-700 mb-2">
                Epochs
                <span className="text-google-blue">{epochs}</span>
              </label>
              <input 
                type="range" min="1" max="50" value={epochs} 
                onChange={(e) => setEpochs(parseInt(e.target.value))}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-google-blue"
                disabled={isTraining}
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Batch Size</label>
              <div className="grid grid-cols-4 gap-2">
                {[16, 32, 64, 128].map(size => (
                  <button
                    key={size}
                    onClick={() => setBatchSize(size)}
                    disabled={isTraining}
                    className={`py-2 text-sm rounded-md border transition-colors
                      ${batchSize === size 
                        ? 'bg-blue-50 border-google-blue text-google-blue font-semibold' 
                        : 'bg-white border-gray-200 text-gray-600 hover:bg-gray-50'}
                    `}
                  >
                    {size}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <button 
            onClick={handleStartTraining}
            disabled={isTraining || dataStatus !== 'ready'}
            className={`mt-8 w-full flex items-center justify-center py-3 rounded-lg text-sm font-bold shadow-sm transition-all
              ${isTraining
                ? 'bg-gray-100 text-gray-400 cursor-wait'
                : 'bg-google-text text-white hover:bg-black'}
               disabled:opacity-50
            `}
          >
            {isTraining ? (
              <><RefreshCw className="animate-spin mr-2" size={18} /> Training in Progress...</>
            ) : (
              <><Play className="mr-2" size={18} /> Start Training</>
            )}
          </button>
        </div>

      </div>

      {/* --- Column 2: Live Monitor --- */}
      <div className="col-span-12 lg:col-span-8 space-y-6">
        
        {/* Loss Chart */}
        <div className="bg-white p-6 rounded-2xl border border-gray-100 shadow-sm h-[400px] flex flex-col">
          <div className="flex justify-between items-center mb-6">
            <div>
              <h2 className="font-semibold text-lg text-gray-800">Training Progress</h2>
              <p className="text-sm text-gray-500">Real-time Loss & Validation Metrics</p>
            </div>
            {isTraining && (
               <div className="flex items-center px-3 py-1 bg-green-50 text-green-700 rounded-full text-xs font-medium animate-pulse">
                 <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
                 Live
               </div>
            )}
          </div>

          <div className="flex-grow w-full">
            {logs.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={logs}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                  <XAxis 
                    dataKey="epoch" 
                    label={{ value: 'Epoch', position: 'insideBottomRight', offset: -5, fontSize: 12 }} 
                    tick={{ fontSize: 12 }}
                  />
                  <YAxis 
                    label={{ value: 'Loss', angle: -90, position: 'insideLeft', fontSize: 12 }} 
                    tick={{ fontSize: 12 }}
                    domain={[0, 'auto']}
                  />
                  <Tooltip 
                     contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}
                  />
                  <Legend verticalAlign="top" height={36} />
                  <Line 
                    type="monotone" dataKey="loss" stroke="#1A73E8" strokeWidth={2} 
                    dot={false} name="Train Loss" animationDuration={300} 
                  />
                  <Line 
                    type="monotone" dataKey="val_loss" stroke="#EA4335" strokeWidth={2} 
                    dot={false} name="Val Loss" animationDuration={300}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-gray-300 border-2 border-dashed border-gray-100 rounded-xl">
                 <Activity size={48} className="mb-2 opacity-20" />
                 <p className="text-sm">Waiting for training to start...</p>
              </div>
            )}
          </div>
        </div>

        {/* Accuracy Chart (Smaller) */}
        <div className="bg-white p-6 rounded-2xl border border-gray-100 shadow-sm h-[300px] flex flex-col">
          <h3 className="font-semibold text-gray-800 mb-4">Accuracy</h3>
          <div className="flex-grow w-full">
             {logs.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={logs}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                  <XAxis dataKey="epoch" hide />
                  <YAxis domain={[0, 1]} tick={{ fontSize: 12 }} />
                  <Tooltip contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }} />
                  <Legend verticalAlign="top" />
                  <Line 
                    type="monotone" dataKey="accuracy" stroke="#34A853" strokeWidth={2} 
                    dot={false} name="Train Acc" animationDuration={300} 
                  />
                  <Line 
                    type="monotone" dataKey="val_accuracy" stroke="#FBBC04" strokeWidth={2} 
                    dot={false} name="Val Acc" animationDuration={300}
                  />
                </LineChart>
              </ResponsiveContainer>
             ) : (
              <div className="h-full flex items-center justify-center text-gray-300">
                <p className="text-xs">No data yet</p>
              </div>
             )}
          </div>
        </div>

      </div>
    </div>
  );
};

// Helper icon
const Activity = ({ size, className }: { size: number, className?: string }) => (
  <svg 
    xmlns="http://www.w3.org/2000/svg" 
    width={size} height={size} viewBox="0 0 24 24" fill="none" 
    stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" 
    className={className}
  >
    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
  </svg>
);

export default TrainingPanel;
