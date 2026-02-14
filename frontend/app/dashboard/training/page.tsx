'use client';

import { useState, useRef, useEffect } from 'react';
import { Play, Activity, Save, Upload, FileText, CheckCircle, AlertCircle, RefreshCw, Database, ChevronRight } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import api from '@/lib/api';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:4000/api';

export default function TrainingPage() {
    // --- DATASET STATE ---
    const [datasets, setDatasets] = useState<any[]>([]);
    const [loadingDatasets, setLoadingDatasets] = useState(true);
    const [selectedDatasetId, setSelectedDatasetId] = useState<number | null>(null);
    const [uploading, setUploading] = useState(false);
    const [datasetError, setDatasetError] = useState('');

    // --- KAGGLE STATE ---
    const [kaggleHandle, setKaggleHandle] = useState('');
    const [importingKaggle, setImportingKaggle] = useState(false);

    // --- TRAINING STATE ---
    const [training, setTraining] = useState(false);
    const [logs, setLogs] = useState<any[]>([]);
    const [finalResult, setFinalResult] = useState<any>(null);
    const [epochs, setEpochs] = useState(10);
    const [modelName, setModelName] = useState('new_model');
    const [trainingError, setTrainingError] = useState('');
    const [initialLoading, setInitialLoading] = useState(true);
    const abortRef = useRef<AbortController | null>(null);

    // Initial load
    useEffect(() => {
        fetchDatasets();
        fetchHistory(); // Keep history for reference

        return () => {
            if (abortRef.current) abortRef.current.abort();
        };
    }, []);

    // --- DATASET FUNCTIONS ---
    const fetchDatasets = async () => {
        try {
            setLoadingDatasets(true);
            const { data } = await api.get('/admin/datasets');
            setDatasets(data);
            // Default select the latest ready dataset if none selected
            if (!selectedDatasetId && data.length > 0) {
                const ready = data.find((d: any) => d.status === 'ready');
                if (ready) setSelectedDatasetId(ready.id);
            }
        } catch (err) {
            console.error("Failed to fetch datasets", err);
            setDatasetError('Failed to load datasets');
        } finally {
            setLoadingDatasets(false);
        }
    };

    const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        if (!e.target.files || !e.target.files[0]) return;
        const file = e.target.files[0];

        setUploading(true);
        setDatasetError('');

        const formData = new FormData();
        formData.append('file', file);

        try {
            await api.post('/admin/datasets/upload', formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });
            fetchDatasets();
        } catch (err: any) {
            setDatasetError(err.response?.data?.error || 'Upload failed');
        } finally {
            setUploading(false);
        }
    };



    const [validationResult, setValidationResult] = useState<any>(null);

    const handleKaggleImport = async () => {
        if (!kaggleHandle) return;
        setImportingKaggle(true);
        setDatasetError('');

        try {
            await api.post('/admin/datasets/kaggle', { handle: kaggleHandle });
            setKaggleHandle('');
            fetchDatasets();
        } catch (err: any) {
            setDatasetError(err.response?.data?.error || err.response?.data?.details || 'Kaggle import failed');
        } finally {
            setImportingKaggle(false);
        }
    };

    const triggerValidation = async (id: number) => {
        try {
            // Clear previous results
            setValidationResult(null);

            const { data } = await api.post(`/admin/datasets/${id}/validate`);
            // data.ai_response contains { valid, checks, report_path }
            setValidationResult(data.ai_response);
            fetchDatasets();
        } catch (err) {
            console.error(err);
            setDatasetError("Validation failed to start.");
        }
    };

    // --- TRAINING FUNCTIONS ---
    const fetchHistory = async () => {
        try {
            const { data } = await api.get('/admin/train/history');
            if (data.history) {
                // ... (Keep existing log formatting logic if needed for history display)
            }
        } catch (err) {
            console.error('Failed to fetch history:', err);
        } finally {
            setInitialLoading(false);
        }
    };

    const startTraining = async () => {
        if (!selectedDatasetId) {
            setTrainingError("Please select a dataset first.");
            return;
        }

        const dataset = datasets.find(d => d.id === selectedDatasetId);
        if (!dataset || dataset.status !== 'ready') {
            setTrainingError("Selected dataset is not ready for training. Please validate it first.");
            return;
        }

        setTraining(true);
        setLogs([]);
        setFinalResult(null);
        setTrainingError('');

        const controller = new AbortController();
        abortRef.current = controller;

        try {
            const token = localStorage.getItem('token');
            const response = await fetch(`${API_URL}/admin/train/stream`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({
                    epochs,
                    batch_size: 32,
                    dataset_path: dataset.path, // Pass the dataset path
                    model_name: modelName       // Pass the model name
                }),
                signal: controller.signal,
            });

            if (!response.ok) throw new Error(`Training failed (${response.status})`);

            const reader = response.body?.getReader();
            if (!reader) throw new Error('Stream not available');

            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            if (data.type === 'complete') {
                                setFinalResult(data.result);
                                setTraining(false);

                                // Save model metadata to backend
                                // We don't await here to avoid blocking the UI update, but we should log errors
                                try {
                                    await api.post('/models', {
                                        name: modelName,
                                        version: '1.0',
                                        accuracy: data.result.test_acc,
                                        datasetId: selectedDatasetId,
                                        path: data.result.model_path
                                    });
                                    console.log("Model saved to database");

                                } catch (e: any) {
                                    console.error("Failed to save model record", e);
                                    const errorDetails = e.response?.data?.details || e.response?.data?.error || e.message;
                                    setTrainingError(`Training finished, but failed to register model: ${errorDetails}`);
                                }
                            } else if (data.type === 'error') {
                                setTrainingError(data.message);
                                setTraining(false);
                            } else {
                                setLogs(prev => [...prev, data]);
                            }
                        } catch { }
                    }
                }
            }
            setTraining(false);
        } catch (err: any) {
            if (err.name !== 'AbortError') {
                setTrainingError(err.message);
            }
            setTraining(false);
        }
    };

    return (
        <div className="space-y-8 pb-10">
            <h1 className="text-3xl font-bold text-gray-800">Data & Training Studio</h1>

            {/* SECTION 1: DATASET PREPARATION */}
            <div className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
                <div className="p-6 border-b border-gray-100 bg-gray-50 flex justify-between items-center">
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-indigo-600 text-white rounded-lg">
                            <Database size={20} />
                        </div>
                        <div>
                            <h2 className="text-lg font-bold text-gray-800">1. Prepare Dataset</h2>
                            <p className="text-sm text-gray-500">Upload, validate, and select data for training</p>
                        </div>
                    </div>
                </div>
            </div>

            <div className="p-6">
                {/* Upload & Kaggle Import Controls */}
                <div className="flex flex-col md:flex-row gap-4 mb-6 p-4 bg-gray-50 rounded-xl border border-gray-100">
                    <div className="flex-1">
                        <label className="block text-xs font-semibold text-gray-500 uppercase mb-2">Option A: Upload Local File</label>
                        <label className={`cursor-pointer w-full flex justify-center items-center gap-2 px-4 py-3 rounded-lg border-2 border-dashed border-gray-300 hover:border-indigo-500 hover:bg-indigo-50 transition text-sm font-medium text-gray-600
                                ${uploading ? 'opacity-50 pointer-events-none' : ''}
                             `}>
                            {uploading ? 'Uploading...' : 'Click to Upload CSV'}
                            <Upload size={18} />
                            <input type="file" accept=".csv" className="hidden" onChange={handleUpload} disabled={uploading} />
                        </label>
                    </div>
                    <div className="hidden md:flex items-center justify-center">
                        <span className="text-gray-400 font-medium bg-white px-2 py-1 rounded border">OR</span>
                    </div>
                    <div className="flex-1">
                        <label className="block text-xs font-semibold text-gray-500 uppercase mb-2">Option B: Import from Kaggle</label>
                        <div className="flex gap-2">
                            <input
                                type="text"
                                placeholder="e.g. sumairaziz/subf-v1-0-dataset..."
                                value={kaggleHandle}
                                onChange={(e) => setKaggleHandle(e.target.value)}
                                className="flex-1 rounded-lg border-gray-300 text-sm focus:border-indigo-500 focus:ring-indigo-500"
                                disabled={importingKaggle}
                            />
                            <button
                                onClick={handleKaggleImport}
                                disabled={!kaggleHandle || importingKaggle}
                                className={`px-4 py-2 rounded-lg font-semibold text-white text-sm whitespace-nowrap transition
                                        ${!kaggleHandle || importingKaggle ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'}
                                    `}
                            >
                                {importingKaggle ? 'Downloading...' : 'Download'}
                            </button>
                        </div>
                    </div>
                </div>

                {datasetError && <div className="mb-4 p-3 bg-red-50 text-red-600 text-sm rounded-lg flex items-center gap-2"><AlertCircle size={16} />{datasetError}</div>}

                {/* Validation Results Panel */}
                {validationResult && (
                    <div className="mb-6 bg-white rounded-xl shadow-lg border border-indigo-100 overflow-hidden animate-in fade-in slide-in-from-top-4">
                        <div className="p-4 bg-indigo-50 border-b border-indigo-100 flex justify-between items-center">
                            <h3 className="font-bold text-indigo-900 flex items-center gap-2">
                                <Activity size={18} className="text-indigo-600" />
                                Data Validation Report
                            </h3>
                            <button onClick={() => setValidationResult(null)} className="text-indigo-400 hover:text-indigo-700">Close</button>
                        </div>
                        <div className="p-5 grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div>
                                <h4 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-3">Schema Checks</h4>
                                <div className="space-y-3">
                                    {validationResult.checks.map((check: any, idx: number) => (
                                        <div key={idx} className="flex items-start gap-3 p-3 rounded-lg border border-gray-100 bg-gray-50">
                                            {check.status === 'Pass' ? (
                                                <CheckCircle size={20} className="text-green-500 mt-0.5" />
                                            ) : check.status === 'Warning' ? (
                                                <AlertCircle size={20} className="text-yellow-500 mt-0.5" />
                                            ) : (
                                                <AlertCircle size={20} className="text-red-500 mt-0.5" />
                                            )}
                                            <div>
                                                <p className="font-semibold text-gray-800 text-sm">{check.name}</p>
                                                <p className="text-xs text-gray-600">{check.detail}</p>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                            <div className="flex flex-col justify-center items-center text-center p-4 border-l border-gray-100">
                                <div className="mb-4">
                                    <h4 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-1">Deep Analysis</h4>
                                    <p className="text-sm text-gray-400">Pandas Profiling Report Generated</p>
                                </div>
                                <a
                                    href={`http://localhost:8000${validationResult.report_path}`}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="px-6 py-3 bg-indigo-600 hover:bg-indigo-700 text-white rounded-full font-bold shadow-md hover:shadow-lg transition flex items-center gap-2"
                                >
                                    <FileText size={18} />
                                    View Full Profiling Report
                                </a>
                                <p className="mt-3 text-xs text-gray-400">Opens in a new tab</p>
                            </div>
                        </div>
                    </div>
                )}


                {loadingDatasets ? (
                    <div className="text-center py-8 text-gray-500">Loading datasets...</div>
                ) : datasets.length === 0 ? (
                    <div className="text-center py-8 text-gray-500 border-2 border-dashed rounded-xl">No datasets found. Upload one to get started.</div>
                ) : (
                    <div className="overflow-x-auto">
                        <table className="w-full text-left">
                            <thead className="bg-gray-50 text-xs font-semibold text-gray-500 uppercase tracking-wider">
                                <tr>
                                    <th className="px-4 py-3">Select</th>
                                    <th className="px-4 py-3">Dataset Name</th>
                                    <th className="px-4 py-3">Status</th>
                                    <th className="px-4 py-3">Rows</th>
                                    <th className="px-4 py-3">Actions</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-gray-100 text-sm">
                                {datasets.map((ds) => (
                                    <tr key={ds.id}
                                        className={`transition cursor-pointer ${selectedDatasetId === ds.id ? 'bg-indigo-50/60' : 'hover:bg-gray-50'}`}
                                        onClick={() => setSelectedDatasetId(ds.id)}
                                    >
                                        <td className="px-4 py-3">
                                            <div className={`w-5 h-5 rounded-full border flex items-center justify-center
                                                    ${selectedDatasetId === ds.id ? 'border-indigo-600 bg-indigo-600' : 'border-gray-300'}
                                                `}>
                                                {selectedDatasetId === ds.id && <div className="w-2 h-2 bg-white rounded-full" />}
                                            </div>
                                        </td>
                                        <td className="px-4 py-3 font-medium text-gray-900 flex items-center gap-2">
                                            <FileText size={16} className="text-gray-400" />
                                            {ds.name}
                                        </td>
                                        <td className="px-4 py-3">
                                            <span className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-medium
                                                    ${ds.status === 'ready' ? 'bg-green-100 text-green-800' :
                                                    ds.status === 'processing' ? 'bg-yellow-100 text-yellow-800' :
                                                        ds.status === 'error' ? 'bg-red-100 text-red-800' :
                                                            'bg-gray-100 text-gray-800'}
                                                `}>
                                                {ds.status === 'processing' && <Activity size={10} className="animate-spin" />}
                                                {ds.status}
                                            </span>
                                        </td>
                                        <td className="px-4 py-3 text-gray-500">{ds.rowCount}</td>
                                        <td className="px-4 py-3">
                                            {ds.status === 'uploaded' && (
                                                <button
                                                    onClick={(e) => { e.stopPropagation(); triggerValidation(ds.id); }}
                                                    className="text-indigo-600 hover:text-indigo-800 font-medium text-xs border border-indigo-200 px-3 py-1 rounded-md hover:bg-indigo-50"
                                                >
                                                    Analyze & Prepare
                                                </button>
                                            )}
                                            {ds.status === 'ready' && (
                                                <span className="text-xs text-green-600 flex items-center gap-1">
                                                    <CheckCircle size={12} /> Ready
                                                </span>
                                            )}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>

            {/* SECTION 2: TRAINING CONFIG */}
            <div className={`transition-opacity duration-500 ${selectedDatasetId ? 'opacity-100' : 'opacity-50 pointer-events-none'}`}>
                <div className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
                    <div className="p-6 border-b border-gray-100 bg-gray-50 flex justify-between items-center">
                        <div className="flex items-center gap-3">
                            <div className="p-2 bg-indigo-600 text-white rounded-lg">
                                <Activity size={20} />
                            </div>
                            <div>
                                <h2 className="text-lg font-bold text-gray-800">2. Train Model</h2>
                                <p className="text-sm text-gray-500">Configure and training your AI model</p>
                            </div>
                        </div>
                    </div>

                    <div className="p-6 grid grid-cols-1 md:grid-cols-3 gap-8 items-end">
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">Model Name</label>
                            <input
                                type="text"
                                value={modelName}
                                onChange={(e) => setModelName(e.target.value)}
                                className="block w-full rounded-lg border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm p-2.5 border"
                                placeholder="e.g. bearing_v1"
                            />
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">Epochs</label>
                            <input
                                type="number"
                                value={epochs}
                                onChange={(e) => setEpochs(Number(e.target.value))}
                                className="block w-full rounded-lg border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm p-2.5 border"
                                min={1} max={100}
                            />
                        </div>
                        <div>
                            <button
                                onClick={startTraining}
                                disabled={training}
                                className={`w-full flex justify-center items-center gap-2 px-6 py-2.5 rounded-lg font-bold text-white transition shadow-lg
                                    ${training ? 'bg-indigo-400 cursor-not-allowed' : 'bg-indigo-600 hover:bg-indigo-700 hover:shadow-indigo-500/30'}
                                `}
                            >
                                {training ? 'Training in Progress...' : 'Start Training Task'}
                                {!training && <Play size={18} />}
                            </button>
                        </div>
                    </div>
                    {trainingError && <div className="mx-6 mb-6 p-3 bg-red-50 text-red-600 text-sm rounded-lg">{trainingError}</div>}
                </div>
            </div>

            {/* SECTION 3: LIVE MONITORING */}
            {
                (training || logs.length > 0) && (
                    <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100 h-[500px] animate-in slide-in-from-bottom-4">
                        <h3 className="text-lg font-semibold mb-6 flex items-center gap-2">
                            <Activity className={training ? "text-green-500 animate-pulse" : "text-gray-400"} />
                            Live Training Metrics
                            {finalResult && <span className="ml-auto text-sm bg-green-100 text-green-700 px-3 py-1 rounded-full font-medium flex items-center gap-1"><CheckCircle size={14} /> Completed</span>}
                        </h3>

                        <ResponsiveContainer width="100%" height="85%">
                            <LineChart data={logs}>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                                <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }} />
                                <YAxis yAxisId="left" orientation="left" stroke="#EA4335" label={{ value: 'Loss', angle: -90, position: 'insideLeft' }} />
                                <YAxis yAxisId="right" orientation="right" stroke="#34A853" domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} label={{ value: 'Accuracy', angle: 90, position: 'insideRight' }} />
                                <Tooltip formatter={(value: any, name: string) => {
                                    if (name.includes('Acc')) return [`${(value * 100).toFixed(2)}%`, name];
                                    return [value.toFixed(4), name];
                                }} contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} />
                                <Legend verticalAlign="top" height={36} iconType="circle" />
                                <Line yAxisId="left" type="monotone" dataKey="loss" stroke="#EA4335" name="Train Loss" strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
                                <Line yAxisId="left" type="monotone" dataKey="val_loss" stroke="#FBBC05" name="Val Loss" strokeWidth={2} dot={false} strokeDasharray="5 5" />
                                <Line yAxisId="right" type="monotone" dataKey="accuracy" stroke="#34A853" name="Train Accuracy" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 6 }} />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                )
            }
        </div >
    );
}
