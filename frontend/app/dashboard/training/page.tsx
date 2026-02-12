'use client';

import { useState, useRef, useEffect } from 'react';
import { Play, Activity, Save } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:4000/api';

export default function TrainingPage() {
    const [training, setTraining] = useState(false);
    const [logs, setLogs] = useState<any[]>([]);
    const [finalResult, setFinalResult] = useState<any>(null);
    const [epochs, setEpochs] = useState(10);
    const [error, setError] = useState('');
    const [initialLoading, setInitialLoading] = useState(true);
    const abortRef = useRef<AbortController | null>(null);

    // Initial load and cleanup
    useEffect(() => {
        fetchHistory();
        return () => {
            if (abortRef.current) {
                abortRef.current.abort();
            }
        };
    }, []);

    const fetchHistory = async () => {
        try {
            setInitialLoading(true);
            const token = typeof window !== 'undefined' ? localStorage.getItem('token') : null;
            const response = await fetch(`${API_URL}/admin/train/history`, {
                headers: {
                    ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
                },
            });

            if (response.ok) {
                const data = await response.json();
                if (data.history) {
                    // Convert history format {acc: [], loss: []} to logs format [{epoch: 1, acc: ...}]
                    const formattedLogs: any[] = [];
                    const keys = Object.keys(data.history);
                    if (keys.length > 0) {
                        const length = data.history[keys[0]].length;
                        for (let i = 0; i < length; i++) {
                            const entry: any = { epoch: i + 1 };
                            keys.forEach(key => {
                                entry[key] = data.history[key][i];
                            });
                            formattedLogs.push(entry);
                        }
                        setLogs(formattedLogs);
                    }
                }
            }
        } catch (err) {
            console.error('Failed to fetch history:', err);
        } finally {
            setInitialLoading(false);
        }
    };

    const startTraining = async () => {
        setTraining(true);
        setLogs([]);
        setFinalResult(null);
        setError('');

        const controller = new AbortController();
        abortRef.current = controller;

        try {
            const token = typeof window !== 'undefined' ? localStorage.getItem('token') : null;
            const response = await fetch(`${API_URL}/admin/train/stream`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
                },
                body: JSON.stringify({ epochs, batch_size: 32 }),
                signal: controller.signal,
            });

            if (!response.ok) {
                if (response.status === 401 || response.status === 403) {
                    setError('Unauthorized. Please log in again.');
                } else {
                    const errData = await response.json().catch(() => ({}));
                    setError(errData.error || errData.detail || `Training failed (status ${response.status})`);
                }
                setTraining(false);
                return;
            }

            const reader = response.body?.getReader();
            if (!reader) {
                setError('Stream not available');
                setTraining(false);
                return;
            }

            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                // Keep the last potentially incomplete line in buffer
                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            if (data.type === 'complete') {
                                setFinalResult(data.result);
                                setTraining(false);
                                // Refresh history to ensure we have the full final stats
                                fetchHistory();
                            } else if (data.type === 'error') {
                                setError(`Training Error: ${data.message}`);
                                setTraining(false);
                            } else {
                                // Epoch progress
                                setLogs(prev => [...prev, data]);
                            }
                        } catch {
                            // Ignore parse errors for keepalive comments etc.
                        }
                    }
                }
            }

            // If we exited the loop without getting a 'complete' event
            setTraining(false);

        } catch (err: any) {
            if (err.name !== 'AbortError') {
                console.error('Training stream error:', err);
                setError(`Connection failed: ${err.message}. Make sure all services are running.`);
            }
            setTraining(false);
        }
    };

    return (
        <div className="space-y-6">
            <h1 className="text-2xl font-bold text-gray-800">Model Training Center</h1>

            {/* Controls */}
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100 flex items-center justify-between">
                <div>
                    <h2 className="text-lg font-semibold">Training Configuration</h2>
                    <p className="text-sm text-gray-500">Configure parameters for the CNN model</p>
                </div>
                <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2">
                        <label className="text-sm font-medium text-gray-600">Epochs:</label>
                        <input
                            type="number"
                            value={epochs}
                            onChange={(e) => setEpochs(Number(e.target.value))}
                            className="w-20 px-3 py-2 border rounded-lg text-center"
                            min={1}
                            max={50}
                        />
                    </div>
                    <button
                        onClick={startTraining}
                        disabled={training}
                        className={`flex items-center gap-2 px-6 py-2 rounded-full font-semibold text-white transition shadow-lg
                    ${training ? 'bg-indigo-400 cursor-not-allowed' : 'bg-indigo-600 hover:bg-indigo-700 hover:shadow-indigo-500/30'}
                `}
                    >
                        {training ? 'Training...' : 'Start Training'}
                        {!training && <Play size={18} />}
                    </button>
                </div>
                {error && <p className="text-red-500 mt-3 text-sm">{error}</p>}
            </div>

            {/* Live Chart */}
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100 h-96">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                    <Activity className="text-indigo-500" />
                    Training Metrics History
                </h3>
                {initialLoading ? (
                    <div className="h-full flex items-center justify-center">
                        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-500"></div>
                    </div>
                ) : (
                    <ResponsiveContainer width="100%" height="85%">
                        <LineChart data={logs}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                            <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }} />
                            <YAxis yAxisId="left" orientation="left" stroke="#EA4335" label={{ value: 'Loss', angle: -90, position: 'insideLeft' }} />
                            <YAxis yAxisId="right" orientation="right" stroke="#34A853" domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} label={{ value: 'Accuracy', angle: 90, position: 'insideRight' }} />
                            <Tooltip formatter={(value: any, name: string) => {
                                if (name.includes('Acc')) return [`${(value * 100).toFixed(2)}%`, name];
                                return [value.toFixed(4), name];
                            }} />
                            <Legend verticalAlign="top" height={36} />
                            <Line yAxisId="left" type="monotone" dataKey="loss" stroke="#EA4335" name="Train Loss" strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
                            <Line yAxisId="left" type="monotone" dataKey="val_loss" stroke="#FBBC05" name="Val Loss" strokeWidth={2} dot={false} strokeDasharray="5 5" />
                            <Line yAxisId="right" type="monotone" dataKey="accuracy" stroke="#34A853" name="Train Accuracy" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 6 }} />
                        </LineChart>
                    </ResponsiveContainer>
                )}
            </div>

            {/* Final Results */}
            {finalResult && (
                <div className="bg-green-50 border border-green-200 p-6 rounded-2xl flex items-center gap-4 animate-in fade-in">
                    <div className="p-3 bg-green-100 rounded-full text-green-600">
                        <Save size={24} />
                    </div>
                    <div>
                        <h4 className="font-bold text-green-800">Training Completed Successfully</h4>
                        <p className="text-green-700">Model saved (Test Accuracy: {(finalResult.test_acc * 100).toFixed(2)}%)</p>
                    </div>
                </div>
            )}
        </div>
    );
}
