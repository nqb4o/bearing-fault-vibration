'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import api from '@/lib/api';
import { AlertCircle, CheckCircle, SmartphoneNfc, Brain, Code, X, Copy, Check } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from 'recharts';

interface ShapExplanation {
    features: string[];
    categories: string[];
    values: number[];
    shap_values: number[];
    base_value: number;
    predicted_class_index: number;
    predicted_class: string;
}

interface DiagnosisResult {
    label: string;
    confidence: number;
    predictions: number[][];
    shap_explanation: ShapExplanation | null;
}

const CATEGORY_COLORS: Record<string, string> = {
    'Time-Domain': '#6366f1',
    'Frequency-Domain': '#f59e0b',
    'Complexity': '#10b981',
    'Time-Frequency': '#ef4444',
};

export default function DiagnosticsPage() {
    const [file, setFile] = useState<File | null>(null);
    const [models, setModels] = useState<any[]>([]);
    const [selectedModel, setSelectedModel] = useState<string>('');
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<DiagnosisResult | null>(null);
    const [error, setError] = useState('');
    const router = useRouter();

    const [showApiModal, setShowApiModal] = useState(false);
    const [snippet, setSnippet] = useState<{ python: string, curl: string } | null>(null);
    const [copied, setCopied] = useState('');

    useEffect(() => {
        if (showApiModal && selectedModel) {
            fetchSnippet(selectedModel);
        }
    }, [showApiModal, selectedModel]);

    const fetchSnippet = async (id: string) => {
        try {
            const { data } = await api.get(`/models/${id}/snippet`);
            setSnippet(data);
        } catch (err) {
            console.error(err);
        }
    };

    const copyToClipboard = (text: string, type: string) => {
        navigator.clipboard.writeText(text);
        setCopied(type);
        setTimeout(() => setCopied(''), 2000);
    };

    useEffect(() => {
        fetchModels();
    }, []);

    const fetchModels = async () => {
        try {
            const { data } = await api.get('/models');
            setModels(data);
            if (data.length > 0) {
                // Determine best default? For now, just pick the latest or none
                setSelectedModel(data[0].id.toString());
            }
        } catch (err) {
            console.error("Failed to fetch models", err);
        }
    };

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
            setResult(null);
            setError('');
        }
    };

    const handleUpload = async () => {
        if (!file) return;
        setLoading(true);
        setError('');

        const formData = new FormData();
        formData.append('file', file);
        if (selectedModel) {
            formData.append('modelId', selectedModel);
        }

        try {
            const { data } = await api.post('/diagnostics/predict', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });
            setResult(data);
        } catch (err: any) {
            const status = err.response?.status;
            if (status === 401 || status === 403) {
                setError('Session expired or unauthorized. Redirecting to login...');
                setTimeout(() => router.push('/login'), 1500);
            } else {
                setError(err.response?.data?.detail || err.response?.data?.error || 'Diagnostic failed. Make sure the AI service is running.');
            }
        } finally {
            setLoading(false);
        }
    };

    // Prepare SHAP chart data: sorted by |shap_value| descending
    const shapChartData = result?.shap_explanation
        ? result.shap_explanation.features
            .map((name, i) => ({
                name,
                category: result.shap_explanation!.categories[i],
                value: result.shap_explanation!.values[i],
                shap: result.shap_explanation!.shap_values[i],
                absShap: Math.abs(result.shap_explanation!.shap_values[i]),
            }))
            .sort((a, b) => b.absShap - a.absShap)
        : [];

    const maxAbsShap = shapChartData.length > 0 ? Math.max(...shapChartData.map(d => d.absShap)) : 1;

    // Custom tooltip for the SHAP bar chart
    const ShapTooltip = ({ active, payload }: any) => {
        if (active && payload && payload.length) {
            const d = payload[0].payload;
            return (
                <div className="bg-white p-3 rounded-lg shadow-lg border border-gray-200 text-sm">
                    <p className="font-bold text-gray-800">{d.name}</p>
                    <p className="text-gray-500 text-xs">{d.category}</p>
                    <div className="mt-1 space-y-0.5">
                        <p>Feature Value: <span className="font-mono font-semibold">{d.value.toFixed(4)}</span></p>
                        <p>SHAP Impact: <span className={`font-mono font-semibold ${d.shap >= 0 ? 'text-red-600' : 'text-blue-600'}`}>
                            {d.shap >= 0 ? '+' : ''}{d.shap.toFixed(4)}
                        </span></p>
                    </div>
                </div>
            );
        }
        return null;
    };

    return (
        <div className="space-y-6">
            <h1 className="text-2xl font-bold text-gray-800">Field Diagnostics</h1>

            {/* Upload Unit */}
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100">
                <h2 className="text-lg font-semibold mb-4">New Analysis</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-4">
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">Select Model</label>
                        <select
                            value={selectedModel}
                            onChange={(e) => setSelectedModel(e.target.value)}
                            className="block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm p-2 border"
                        >
                            <option value="">-- Default Model --</option>
                            {models.map((m) => (
                                <option key={m.id} value={m.id}>
                                    {m.name} (v{m.version}) - {(m.accuracy * 100).toFixed(1)}% Acc
                                </option>
                            ))}
                        </select>
                        {selectedModel && (
                            <button
                                onClick={() => setShowApiModal(true)}
                                className="text-xs text-indigo-600 font-medium mt-1 hover:underline flex items-center gap-1"
                            >
                                <Code size={12} />
                                Get API Code for this model
                            </button>
                        )}
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">Upload Sensor Data</label>
                        <input
                            type="file"
                            accept=".csv"
                            onChange={handleFileChange}
                            className="block w-full text-sm text-gray-500
                                file:mr-4 file:py-2 file:px-4
                                file:rounded-full file:border-0
                                file:text-sm file:font-semibold
                                file:bg-blue-50 file:text-blue-700
                                hover:file:bg-blue-100 placeholder-gray-400"
                        />
                    </div>
                </div>

                <div className="flex justify-end">
                    <button
                        onClick={handleUpload}
                        disabled={!file || loading}
                        className={`flex items-center gap-2 px-6 py-2 rounded-full font-semibold text-white transition
                            ${!file || loading ? 'bg-gray-300 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'}
                        `}
                    >
                        {loading ? 'Processing...' : 'Run Diagnostics'}
                        {!loading && <SmartphoneNfc size={18} />}
                    </button>
                </div>
                {error && <p className="text-red-500 mt-2 text-sm">{error}</p>}
            </div>

            {/* Results */}
            {result && (
                <div className="space-y-6 animate-in slide-in-from-bottom-4 fade-in duration-500">
                    {/* Status Card */}
                    <div className={`p-6 rounded-2xl shadow-sm border-l-8 ${result.label === 'Normal' ? 'border-green-500 bg-green-50' : 'border-red-500 bg-red-50'}`}>
                        <div className="flex items-start justify-between">
                            <div>
                                <p className="text-sm font-medium text-gray-500 uppercase tracking-wider">System Status</p>
                                <h3 className={`text-3xl font-bold mt-1 ${result.label === 'Normal' ? 'text-green-700' : 'text-red-700'}`}>
                                    {result.label.toUpperCase()}
                                </h3>
                            </div>
                            {result.label === 'Normal' ? <CheckCircle size={40} className="text-green-500" /> : <AlertCircle size={40} className="text-red-500" />}
                        </div>
                        <div className="mt-4 flex items-center gap-2">
                            <span className="text-sm text-gray-600">AI Confidence:</span>
                            <div className="h-2 flex-1 bg-gray-200 rounded-full overflow-hidden">
                                <div className="h-full bg-current opacity-80" style={{ width: `${result.confidence * 100}%` }} />
                            </div>
                            <span className="text-sm font-bold">{(result.confidence * 100).toFixed(1)}%</span>
                        </div>
                    </div>

                    {/* SHAP Explanation Chart */}
                    {result.shap_explanation && shapChartData.length > 0 && (
                        <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100">
                            <div className="flex items-center justify-between mb-2">
                                <div>
                                    <h3 className="text-lg font-semibold flex items-center gap-2">
                                        <Brain className="text-indigo-500" size={22} />
                                        SHAP Feature Explanation
                                    </h3>
                                    <p className="text-xs text-gray-400 mt-1">
                                        How each vibration feature contributes to the "{result.shap_explanation.predicted_class}" diagnosis.
                                        <span className="text-red-500 font-medium"> Red</span> = pushes toward this diagnosis,
                                        <span className="text-blue-500 font-medium"> Blue</span> = pushes against.
                                    </p>
                                </div>
                            </div>

                            {/* Category Legend */}
                            <div className="flex flex-wrap gap-3 mb-4 mt-3">
                                {Object.entries(CATEGORY_COLORS).map(([cat, color]) => (
                                    <div key={cat} className="flex items-center gap-1.5 text-xs">
                                        <span className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: color, opacity: 0.7 }} />
                                        <span className="text-gray-600">{cat}</span>
                                    </div>
                                ))}
                            </div>

                            {/* Horizontal Bar Chart */}
                            <div style={{ height: `${Math.max(shapChartData.length * 32, 300)}px` }}>
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart
                                        data={shapChartData}
                                        layout="vertical"
                                        margin={{ top: 5, right: 40, left: 140, bottom: 5 }}
                                    >
                                        <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                                        <XAxis
                                            type="number"
                                            domain={[-maxAbsShap * 1.1, maxAbsShap * 1.1]}
                                            tickFormatter={(v: number) => v.toFixed(3)}
                                            label={{ value: 'SHAP Value (impact on prediction)', position: 'bottom', offset: 0, style: { fontSize: 11, fill: '#9ca3af' } }}
                                        />
                                        <YAxis
                                            type="category"
                                            dataKey="name"
                                            width={130}
                                            tick={{ fontSize: 12 }}
                                        />
                                        <Tooltip content={<ShapTooltip />} />
                                        <ReferenceLine x={0} stroke="#94a3b8" strokeWidth={1.5} />
                                        <Bar dataKey="shap" radius={[0, 4, 4, 0]} barSize={18}>
                                            {shapChartData.map((entry, index) => (
                                                <Cell
                                                    key={`cell-${index}`}
                                                    fill={entry.shap >= 0 ? '#ef4444' : '#3b82f6'}
                                                    fillOpacity={0.75 + 0.25 * (entry.absShap / maxAbsShap)}
                                                />
                                            ))}
                                        </Bar>
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>

                            {/* Feature Details Table */}
                            <details className="mt-4">
                                <summary className="text-sm text-gray-500 cursor-pointer hover:text-gray-700 font-medium">
                                    View All Feature Values
                                </summary>
                                <div className="overflow-x-auto mt-2">
                                    <table className="w-full text-sm text-left">
                                        <thead>
                                            <tr className="border-b border-gray-200">
                                                <th className="py-2 pr-4 text-gray-500 font-medium">Feature</th>
                                                <th className="py-2 pr-4 text-gray-500 font-medium">Category</th>
                                                <th className="py-2 pr-4 text-gray-500 font-medium text-right">Value</th>
                                                <th className="py-2 text-gray-500 font-medium text-right">SHAP Impact</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {shapChartData.map((d, i) => (
                                                <tr key={i} className="border-b border-gray-50 hover:bg-gray-50">
                                                    <td className="py-1.5 pr-4 font-medium">{d.name}</td>
                                                    <td className="py-1.5 pr-4">
                                                        <span
                                                            className="text-xs px-2 py-0.5 rounded-full text-white"
                                                            style={{ backgroundColor: CATEGORY_COLORS[d.category] || '#6b7280' }}
                                                        >
                                                            {d.category}
                                                        </span>
                                                    </td>
                                                    <td className="py-1.5 pr-4 text-right font-mono text-gray-700">{d.value.toFixed(4)}</td>
                                                    <td className={`py-1.5 text-right font-mono font-semibold ${d.shap >= 0 ? 'text-red-600' : 'text-blue-600'}`}>
                                                        {d.shap >= 0 ? '+' : ''}{d.shap.toFixed(4)}
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            </details>
                        </div>
                    )}

                    {/* Fallback: no SHAP explanation */}
                    {!result.shap_explanation && (
                        <div className="bg-amber-50 border border-amber-200 p-4 rounded-2xl text-sm text-amber-800">
                            <p className="font-semibold">SHAP Explanation Unavailable</p>
                            <p className="mt-1">The feature model is not loaded. Please train the model first to enable SHAP feature explanations.</p>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
