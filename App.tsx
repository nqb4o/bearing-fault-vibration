import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import FileUpload from './components/FileUpload';
import HealthGauge from './components/HealthGauge';
import FeatureMap from './components/FeatureMap';
import TrainingPanel from './components/TrainingPanel'; // Import the new Admin Panel
import { AppState, AnalysisResult, DiagnosisStatus, ViewMode } from './types';
import { uploadAndAnalyze } from './services/mockApiService';
import { Bell, Search, ShieldCheck } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const App: React.FC = () => {
  // Navigation State
  const [currentView, setCurrentView] = useState<ViewMode>(ViewMode.DIAGNOSTICS);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);

  // Diagnostics State
  const [appState, setAppState] = useState<AppState>(AppState.IDLE);
  const [result, setResult] = useState<AnalysisResult | null>(null);

  const handleUpload = async (file: File) => {
    setAppState(AppState.PROCESSING);
    try {
      const data = await uploadAndAnalyze(file);
      setResult(data);
      setAppState(AppState.COMPLETE);
    } catch (error) {
      console.error(error);
      setAppState(AppState.ERROR);
    }
  };

  const resetAnalysis = () => {
    setAppState(AppState.IDLE);
    setResult(null);
  }

  // Render Logic
  const renderContent = () => {
    if (currentView === ViewMode.TRAINING) {
      return (
        <div className="animate-in fade-in duration-500">
          <div className="mb-8">
            <h1 className="text-2xl font-bold text-gray-800 mb-1">Model Training & Data Pipeline</h1>
            <p className="text-gray-500">Admin dashboard for retraining the CNN model and updating reference statistics.</p>
          </div>
          <TrainingPanel />
        </div>
      );
    }

    // Default: Diagnostics View
    return (
      <div className="animate-in fade-in duration-500">
          {/* Welcome / Action Section */}
          <div className="mb-8">
            <h1 className="text-2xl font-bold text-gray-800 mb-1">Field Diagnostics</h1>
            <p className="text-gray-500">Upload vibration data (CSV) to detect bearing faults using AI.</p>
          </div>

          <div className="grid grid-cols-12 gap-6">
            
            {/* Left Col: Upload & Diagnostic Result */}
            <div className="col-span-12 lg:col-span-8 space-y-6">
              {/* Upload Card */}
              <div className="bg-white p-6 rounded-2xl border border-gray-100 shadow-sm">
                 <div className="flex justify-between items-center mb-4">
                   <h2 className="text-lg font-semibold text-gray-800">New Analysis</h2>
                   {appState === AppState.COMPLETE && (
                     <button 
                      onClick={resetAnalysis}
                      className="text-sm text-google-blue font-medium hover:underline"
                     >
                       Reset Analysis
                     </button>
                   )}
                 </div>
                 <FileUpload onUpload={handleUpload} appState={appState} />
              </div>

              {/* Diagnostic Result (Only when complete) */}
              {appState === AppState.COMPLETE && result && (
                <div className="bg-white p-6 rounded-2xl border border-gray-100 shadow-sm space-y-6">
                    {/* Alert Header */}
                    <div className={`p-4 rounded-xl border-l-4 ${
                        result.status === DiagnosisStatus.NORMAL 
                            ? 'bg-green-50 border-green-500 text-green-900' 
                            : result.status === DiagnosisStatus.INNER_RACE_FAULT
                              ? 'bg-red-50 border-red-500 text-red-900'
                              : 'bg-yellow-50 border-yellow-500 text-yellow-900'
                    }`}>
                        <div className="flex items-center gap-2 font-bold text-lg">
                            {result.status === DiagnosisStatus.NORMAL 
                              ? '🟢 SYSTEM NORMAL' 
                              : `🔴 CRITICAL FAULT: ${result.status.toUpperCase()}`}
                            <span className="text-sm font-normal opacity-80">(Confidence: {(result.confidence * 100).toFixed(1)}%)</span>
                        </div>
                        <div className="flex items-center gap-2 mt-2 text-sm font-medium opacity-90">
                            {result.status === DiagnosisStatus.NORMAL 
                                ? "System operating within normal parameters." 
                                : "⚠️ Recommendation: Stop machine and inspect bearing immediately."}
                        </div>
                    </div>

                    {/* Probability Chart */}
                    <div>
                        <div className="flex items-center gap-2 mb-4">
                            <span className="text-xl">📊</span>
                            <h3 className="font-semibold text-gray-800">Detailed Probability Analysis</h3>
                        </div>
                        <div className="h-[300px] w-full bg-slate-50 rounded-lg p-4 border border-slate-100">
                             <ResponsiveContainer width="100%" height="100%">
                                <BarChart layout="vertical" data={result.probabilities} margin={{ top: 0, right: 30, left: 30, bottom: 0 }}>
                                    <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} stroke="#e2e8f0" />
                                    <XAxis type="number" domain={[0, 1]} tickFormatter={(val) => `${(val * 100).toFixed(0)}%`} stroke="#94a3b8" fontSize={12} />
                                    <YAxis dataKey="label" type="category" width={120} tick={{fontSize: 12, fill: '#475569', fontWeight: 500}} axisLine={false} tickLine={false} />
                                    <Tooltip 
                                        cursor={{fill: 'rgba(0,0,0,0.05)'}}
                                        contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }}
                                        formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, 'Probability']}
                                    />
                                    <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={24}>
                                        {result.probabilities.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={
                                                entry.label === DiagnosisStatus.NORMAL ? '#34A853' : 
                                                entry.label === DiagnosisStatus.INNER_RACE_FAULT ? '#EA4335' :
                                                '#FBBC04'
                                            } />
                                        ))}
                                    </Bar>
                                </BarChart>
                             </ResponsiveContainer>
                        </div>
                    </div>
                </div>
              )}
              
              {/* Skeleton Loading for Result */}
              {appState === AppState.PROCESSING && (
                <div className="bg-white p-6 rounded-2xl border border-gray-100 shadow-sm h-[250px] animate-pulse">
                  <div className="h-4 bg-gray-200 rounded w-1/4 mb-4"></div>
                  <div className="h-20 bg-gray-100 rounded-lg mb-6"></div>
                  <div className="h-32 bg-gray-100 rounded-lg"></div>
                </div>
              )}
            </div>

            {/* Right Col: Diagnostics Result */}
            <div className="col-span-12 lg:col-span-4 space-y-6">
              
              {/* Health Gauge */}
              {appState === AppState.COMPLETE && result ? (
                <HealthGauge score={result.confidence} status={result.status} />
              ) : appState === AppState.PROCESSING ? (
                 <div className="h-[280px] bg-white rounded-2xl border border-gray-100 shadow-sm animate-pulse p-6 flex flex-col items-center justify-center">
                    <div className="w-40 h-40 rounded-full border-8 border-gray-100"></div>
                 </div>
              ) : (
                <div className="h-[280px] bg-white rounded-2xl border border-gray-100 shadow-sm flex flex-col items-center justify-center text-center p-6 text-gray-400">
                  <ShieldCheck size={48} className="mb-2 opacity-20" />
                  <p>Awaiting Data...</p>
                </div>
              )}

              {/* Feature Map */}
              <div className="h-[400px]">
                <FeatureMap currentResult={result} />
              </div>

              {/* Quick Summary Card */}
              {appState === AppState.COMPLETE && result && (
                <div className={`p-4 rounded-xl border-l-4 shadow-sm bg-white
                  ${result.status === DiagnosisStatus.NORMAL ? 'border-green-500' : result.status === DiagnosisStatus.INNER_RACE_FAULT ? 'border-red-500' : 'border-yellow-500'}
                `}>
                  <h4 className="font-semibold text-gray-800 text-sm">Diagnostic Summary</h4>
                  <p className="text-sm text-gray-600 mt-1">
                    {result.status === DiagnosisStatus.NORMAL 
                      ? "System operating within normal parameters. No maintenance required."
                      : `Anomaly detected matching ${result.status} signature. Recommend scheduling inspection.`
                    }
                  </p>
                </div>
              )}
            </div>
          </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-slate-50 font-sans">
      <Sidebar 
        currentView={currentView} 
        onViewChange={setCurrentView}
        isCollapsed={isSidebarCollapsed}
        onToggle={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
      />

      {/* Main Content Area */}
      <main className={`${isSidebarCollapsed ? 'md:ml-20' : 'md:ml-64'} min-h-screen flex flex-col transition-all duration-300 ease-in-out`}>
        
        {/* Top Header */}
        <header className="h-16 bg-white border-b border-gray-200 flex items-center justify-between px-8 sticky top-0 z-40">
          <div className="flex items-center bg-gray-100 rounded-lg px-3 py-2 w-96">
            <Search size={18} className="text-gray-400 mr-2" />
            <input 
              type="text" 
              placeholder="Search assets, sensors, or dates..." 
              className="bg-transparent border-none outline-none text-sm w-full text-gray-700 placeholder-gray-400"
            />
          </div>
          <div className="flex items-center space-x-4">
            <button className="p-2 text-gray-500 hover:bg-gray-100 rounded-full relative">
              <Bell size={20} />
              <span className="absolute top-2 right-2 w-2 h-2 bg-red-500 rounded-full border border-white"></span>
            </button>
            <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-white font-medium">
              JD
            </div>
          </div>
        </header>

        {/* Dashboard Content */}
        <div className="flex-1 p-8 max-w-7xl mx-auto w-full">
           {renderContent()}
        </div>
      </main>
    </div>
  );
};

export default App;