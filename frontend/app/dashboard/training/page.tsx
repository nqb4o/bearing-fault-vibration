'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Database, Activity, Play, CheckCircle, AlertTriangle, Upload, Loader2, FileText, X, AlertCircle, Info } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import api, { API_URL } from '@/lib/api';

export default function TrainingPage() {
    // --- STATE ---
    const [datasets, setDatasets] = useState<any[]>([]);
    const [selectedDatasetId, setSelectedDatasetId] = useState<number | null>(null);
    const [datasetError, setDatasetError] = useState('');
    const [uploading, setUploading] = useState(false);
    const [loadingDatasets, setLoadingDatasets] = useState(true);

    // Kaggle
    const [kaggleHandle, setKaggleHandle] = useState('');
    const [importingKaggle, setImportingKaggle] = useState(false);

    // Validation / Details
    const [viewingDataset, setViewingDataset] = useState<any>(null); // State for the detail modal

    // Training
    const [epochs, setEpochs] = useState(10);
    const [modelName, setModelName] = useState('new_model');
    const [training, setTraining] = useState(false);
    const [logs, setLogs] = useState<any[]>([]);
    const [finalResult, setFinalResult] = useState<any>(null);
    const [trainingError, setTrainingError] = useState('');

    // Initial Load
    // const [initialLoading, setInitialLoading] = useState(true); // Unused for now

    const abortRef = useRef<AbortController | null>(null);

    useEffect(() => {
        fetchDatasets();
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

            // Default select ready one
            if (!selectedDatasetId && data.length > 0) {
                const ready = data.find((d: any) => d.status === 'ready');
                if (ready) setSelectedDatasetId(ready.id);
            }
        } catch (err) {
            console.error('Failed to fetch datasets:', err);
            setDatasetError('Failed to load datasets');
        } finally {
            setLoadingDatasets(false);
        }
    };

    const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        if (!e.target.files || e.target.files.length === 0) return;

        const files = Array.from(e.target.files);
        setUploading(true);
        setDatasetError('');

        const formData = new FormData();
        files.forEach(file => {
            formData.append('files', file);
        });

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
            // setValidationResult(null); // No longer needed
            const { data } = await api.post(`/admin/datasets/${id}/validate`);
            // Update local dataset list to reflect new status/result
            setDatasets(prev => prev.map(d => d.id === id ? { ...d, status: 'ready', validationResult: data.ai_response } : d));
            setViewingDataset({ ...datasets.find(d => d.id === id), validationResult: data.ai_response });
        } catch (err) {
            console.error(err);
            setDatasetError("Validation failed to start.");
        }
    };

    const viewDetails = (dataset: any) => {
        setViewingDataset(dataset);
    };

    const openReport = (reportPath: string) => {
        if (!reportPath) return;
        // Construct full URL. AI Service (Port 8000) serves reports, Backend (Port 4000) matches API_URL.
        // We need to replace the Backend URL with the AI Service URL.
        // If API_URL is http://localhost:4000/api -> http://localhost:8000

        let baseUrl = API_URL.replace('/api', '');
        // Heuristic: if localhost:4000, switch to 8000. If domain, maybe differ? 
        // For now, let's just do a string replacement if it contains 4000.
        // Or better, let's just assume simple replacement for now as per user request.

        if (baseUrl.includes(':4000')) {
            baseUrl = baseUrl.replace(':4000', ':8000');
        }

        const url = `${baseUrl}${reportPath}`;
        window.open(url, '_blank');
    };

    // --- TRAINING FUNCTIONS ---
    const startTraining = async () => {
        if (!selectedDatasetId) {
            setTrainingError("Please select a dataset first.");
            return;
        }

        const dataset = datasets.find(d => d.id === selectedDatasetId);
        if (!dataset || dataset.status !== 'ready') {
            setTrainingError("Selected dataset is not ready for training.");
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
                    dataset_path: dataset.path,
                    model_name: modelName
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

                                // Save model logic
                                try {
                                    await api.post('/models', {
                                        name: modelName,
                                        version: '1.0',
                                        accuracy: data.result.test_acc,
                                        datasetId: selectedDatasetId,
                                        path: data.result.model_path
                                    });
                                } catch (e: any) {
                                    console.error("Failed to save model", e);
                                    setTrainingError(`Training finished but failed to save model: ${e.message}`);
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

                <div className="p-6">
                    {/* Upload Controls */}
                    <div className="flex flex-col md:flex-row gap-4 mb-6 p-4 bg-gray-50 rounded-xl border border-gray-100">
                        <div className="flex-1">
                            <label className="block text-xs font-semibold text-gray-500 uppercase mb-2">Option A: Upload Local File</label>
                            <label className={`cursor-pointer w-full flex justify-center items-center gap-2 px-4 py-3 rounded-lg border-2 border-dashed border-gray-300 hover:border-indigo-500 hover:bg-indigo-50 transition text-sm font-medium text-gray-600 ${uploading ? 'opacity-50 pointer-events-none' : ''}`}>
                                {uploading ? 'Uploading...' : 'Click to Upload CSVs (Select Multiple)'}
                                <Upload size={18} />
                                <input type="file" accept=".csv" multiple className="hidden" onChange={handleUpload} disabled={uploading} />
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
                                    placeholder="e.g. sumairaziz/subf-v1-0-dataset"
                                    value={kaggleHandle}
                                    onChange={(e) => setKaggleHandle(e.target.value)}
                                    className="flex-1 rounded-lg border-gray-300 text-sm focus:border-indigo-500 focus:ring-indigo-500 px-4 py-2 border"
                                    disabled={importingKaggle}
                                />
                                <button
                                    onClick={handleKaggleImport}
                                    disabled={!kaggleHandle || importingKaggle}
                                    className={`px-4 py-2 rounded-lg font-semibold text-white text-sm whitespace-nowrap transition ${!kaggleHandle || importingKaggle ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'}`}
                                >
                                    {importingKaggle ? 'Downloading...' : 'Download'}
                                </button>
                            </div>
                        </div>
                    </div>

                    {datasetError && <div className="mb-4 p-3 bg-red-50 text-red-600 text-sm rounded-lg flex items-center gap-2"><AlertCircle size={16} />{datasetError}</div>}

                    {/* Dataset List */}
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
                                        <tr key={ds.id} className={`transition cursor-pointer ${selectedDatasetId === ds.id ? 'bg-indigo-50/60' : 'hover:bg-gray-50'}`} onClick={() => setSelectedDatasetId(ds.id)}>
                                            <td className="px-4 py-3">
                                                <div className={`w-5 h-5 rounded-full border flex items-center justify-center ${selectedDatasetId === ds.id ? 'border-indigo-600 bg-indigo-600' : 'border-gray-300'}`}>
                                                    {selectedDatasetId === ds.id && <div className="w-2 h-2 bg-white rounded-full" />}
                                                </div>
                                            </td>
                                            <td className="px-4 py-3 font-medium text-gray-900 flex items-center gap-2">
                                                <Database size={16} className="text-gray-400" />
                                                {ds.name}
                                            </td>
                                            <td className="px-4 py-3">
                                                <span className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-medium ${ds.status === 'ready' ? 'bg-green-100 text-green-800' : ds.status === 'processing' ? 'bg-yellow-100 text-yellow-800' : ds.status === 'error' ? 'bg-red-100 text-red-800' : 'bg-gray-100 text-gray-800'}`}>
                                                    {ds.status === 'processing' && <Loader2 size={10} className="animate-spin" />}
                                                    {ds.status}
                                                </span>
                                            </td>
                                            <td className="px-4 py-3 text-gray-500">{ds.rowCount}</td>
                                            <td className="px-4 py-3">
                                                <div className="flex gap-2">
                                                    {ds.status === 'ready' ? (
                                                        <>
                                                            <button onClick={(e) => { e.stopPropagation(); setSelectedDatasetId(ds.id); }} className={`px-4 py-2 rounded-lg text-sm font-medium transition flex items-center gap-2 ${selectedDatasetId === ds.id ? 'bg-green-600 text-white shadow-md' : 'bg-white border border-gray-200 text-gray-700 hover:bg-gray-50'}`}>
                                                                {selectedDatasetId === ds.id ? <CheckCircle size={16} /> : <div className="w-4 h-4 rounded-full border border-gray-400" />}
                                                                {selectedDatasetId === ds.id ? 'Selected' : 'Select'}
                                                            </button>
                                                        </>
                                                    ) : (
                                                        <button onClick={(e) => { e.stopPropagation(); triggerValidation(ds.id); }} disabled={ds.status === 'processing'} className="px-4 py-2 bg-indigo-600 text-white rounded-lg text-sm font-medium hover:bg-indigo-700 transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2">
                                                            {ds.status === 'processing' ? <Loader2 className="animate-spin" size={16} /> : <Play size={16} />}
                                                            Validate & Profile
                                                        </button>
                                                    )}

                                                    {/* ALWAYS SHOW DETAIL BUTTON */}
                                                    <button onClick={(e) => { e.stopPropagation(); viewDetails(ds); }} className="p-2 text-gray-500 hover:text-indigo-600 bg-gray-50 hover:bg-indigo-50 rounded-lg transition" title="View Dataset Details">
                                                        <Info size={18} />
                                                    </button>
                                                </div>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}
                </div>
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
                            <input type="text" value={modelName} onChange={(e) => setModelName(e.target.value)} className="block w-full rounded-lg border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm p-2.5 border" placeholder="e.g. bearing_v1" />
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">Epochs</label>
                            <input type="number" value={epochs} onChange={(e) => setEpochs(Number(e.target.value))} className="block w-full rounded-lg border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm p-2.5 border" min={1} max={100} />
                        </div>
                        <div>
                            <button onClick={startTraining} disabled={training} className={`w-full flex justify-center items-center gap-2 px-6 py-2.5 rounded-lg font-bold text-white transition shadow-lg ${training ? 'bg-indigo-400 cursor-not-allowed' : 'bg-indigo-600 hover:bg-indigo-700 hover:shadow-indigo-500/30'}`}>
                                {training ? 'Training in Progress...' : 'Start Training Task'}
                                {!training && <Play size={18} />}
                            </button>
                        </div>
                    </div>
                    {trainingError && <div className="mx-6 mb-6 p-3 bg-red-50 text-red-600 text-sm rounded-lg">{trainingError}</div>}
                </div>
            </div>

            {/* SECTION 3: LIVE MONITORING */}
            {(training || logs.length > 0) && (
                <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100 h-[500px]">
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
            )}

            {/* DATASET DETAILS MODAL */}
            {viewingDataset && (
                <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4 animate-in fade-in duration-200">
                    <div className="bg-white rounded-2xl shadow-xl w-full max-w-2xl max-h-[90vh] overflow-y-auto transform scale-100 transition-all">
                        <div className="p-6 border-b border-gray-100 flex justify-between items-center sticky top-0 bg-white z-10">
                            <h3 className="text-xl font-bold flex items-center gap-2">
                                <Database className="text-indigo-600" />
                                Dataset Details: {viewingDataset.name}
                            </h3>
                            <button onClick={() => setViewingDataset(null)} className="p-2 hover:bg-gray-100 rounded-lg transition">
                                <X size={20} />
                            </button>
                        </div>
                        <div className="p-6 space-y-6">
                            {/* Basic Info */}
                            <div className="grid grid-cols-2 gap-4">
                                <div className="p-4 bg-gray-50 rounded-xl border border-gray-100">
                                    <p className="text-xs text-gray-500 uppercase font-semibold">Row Count</p>
                                    <p className="text-lg font-bold text-gray-800">{viewingDataset.rowCount?.toLocaleString()}</p>
                                </div>
                                <div className="p-4 bg-gray-50 rounded-xl border border-gray-100">
                                    <p className="text-xs text-gray-500 uppercase font-semibold">Size</p>
                                    <p className="text-lg font-bold text-gray-800">{(viewingDataset.size / 1024 / 1024).toFixed(2)} MB</p>
                                </div>
                            </div>

                            {/* Validation Status */}
                            {viewingDataset.validationResult ? (
                                <div className="space-y-4">
                                    <h4 className="font-bold text-gray-800 flex items-center gap-2 border-b pb-2">
                                        <Activity size={18} /> Validation Results
                                    </h4>

                                    <div className={`p-4 rounded-xl border ${viewingDataset.validationResult.valid ? 'bg-green-50 border-green-200 text-green-800' : 'bg-red-50 border-red-200 text-red-800'}`}>
                                        <h4 className="font-bold flex items-center gap-2">
                                            {viewingDataset.validationResult.valid ? <CheckCircle size={20} /> : <AlertTriangle size={20} />}
                                            Status: {viewingDataset.validationResult.valid ? 'Valid Dataset' : 'Issues Found'}
                                        </h4>
                                    </div>

                                    <div className="space-y-2">
                                        {viewingDataset.validationResult.checks?.map((check: any, idx: number) => (
                                            <div key={idx} className="flex items-start gap-3 p-3 bg-gray-50 rounded-lg border border-gray-100">
                                                <div className={`mt-1 ${check.status === 'Pass' ? 'text-green-600' : check.status === 'Warning' ? 'text-yellow-600' : 'text-red-600'}`}>
                                                    {check.status === 'Pass' ? <CheckCircle size={16} /> : <AlertTriangle size={16} />}
                                                </div>
                                                <div>
                                                    <div className="font-medium text-gray-900">{check.name}</div>
                                                    <div className="text-sm text-gray-600">{check.detail}</div>
                                                </div>
                                            </div>
                                        ))}
                                    </div>

                                    {viewingDataset.validationResult.report_path && (
                                        <div className="mt-4">
                                            <button
                                                onClick={() => openReport(viewingDataset.validationResult.report_path)}
                                                className="w-full py-3 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl font-medium flex justify-center items-center gap-2 transition shadow-sm hover:shadow"
                                            >
                                                <FileText size={18} />
                                                Open Full Profiling Report
                                            </button>
                                        </div>
                                    )}
                                </div>
                            ) : (
                                <div className="p-8 text-center bg-gray-50 rounded-xl border border-dashed border-gray-300">
                                    <AlertCircle className="mx-auto text-gray-400 mb-2" size={32} />
                                    <p className="text-gray-500 font-medium">No validation results available for this dataset.</p>
                                    <p className="text-sm text-gray-400 mt-1">Run validation to generate profiling reports and checks.</p>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
