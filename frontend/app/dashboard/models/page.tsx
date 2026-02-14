'use client';

import { useState, useEffect } from 'react';
import { Box, Code, Copy, Check } from 'lucide-react';
import api from '@/lib/api';

export default function ModelsPage() {
    const [models, setModels] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);
    const [selectedModel, setSelectedModel] = useState<number | null>(null);
    const [snippets, setSnippets] = useState<{ python: string; curl: string } | null>(null);
    const [copied, setCopied] = useState('');

    useEffect(() => {
        fetchModels();
    }, []);

    const fetchModels = async () => {
        try {
            const { data } = await api.get('/models');
            setModels(data);
        } catch (err) {
            console.error('Failed to fetch models', err);
        } finally {
            setLoading(false);
        }
    };

    const handleSelectModel = async (id: number) => {
        setSelectedModel(id);
        // Fetch snippet
        try {
            const { data } = await api.get(`/models/${id}/snippet`);
            setSnippets(data);
        } catch (err) {
            console.error(err);
        }
    };

    const copyToClipboard = async (text: string, type: string) => {
        try {
            if (navigator?.clipboard?.writeText) {
                await navigator.clipboard.writeText(text);
            } else {
                // Fallback for non-secure contexts or older browsers
                const textArea = document.createElement("textarea");
                textArea.value = text;
                textArea.style.position = "fixed";  // Avoid scrolling to bottom
                document.body.appendChild(textArea);
                textArea.focus();
                textArea.select();
                try {
                    document.execCommand('copy');
                } catch (err) {
                    console.error('Fallback: Oops, unable to copy', err);
                }
                document.body.removeChild(textArea);
            }
            setCopied(type);
            setTimeout(() => setCopied(''), 2000);
        } catch (err) {
            console.error('Failed to copy!', err);
        }
    };

    return (
        <div className="space-y-6">
            <h1 className="text-2xl font-bold text-gray-800">Model Marketplace</h1>
            <p className="text-gray-600">Select a model to use for your predictions or integrate it via API.</p>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Models List */}
                <div className="lg:col-span-2 space-y-4">
                    {loading ? (
                        <div className="text-center p-10 text-gray-500">Loading models...</div>
                    ) : models.length === 0 ? (
                        <div className="text-center p-10 bg-white rounded-2xl border border-gray-100 text-gray-500">
                            No models available. Ask an admin to train one.
                        </div>
                    ) : (
                        models.map((model) => (
                            <div
                                key={model.id}
                                onClick={() => handleSelectModel(model.id)}
                                className={`bg-white p-6 rounded-2xl border cursor-pointer transition relative overflow-hidden group
                                    ${selectedModel === model.id ? 'border-indigo-500 ring-2 ring-indigo-500/20 shadow-lg' : 'border-gray-100 hover:border-indigo-300 hover:shadow-md'}
                                `}
                            >
                                <div className="flex justify-between items-start">
                                    <div className="flex gap-4">
                                        <div className={`p-3 rounded-xl ${selectedModel === model.id ? 'bg-indigo-600 text-white' : 'bg-gray-100 text-gray-600 group-hover:bg-indigo-50 group-hover:text-indigo-600'}`}>
                                            <Box size={24} />
                                        </div>
                                        <div>
                                            <h3 className="font-semibold text-lg text-gray-800">{model.name}</h3>
                                            <p className="text-sm text-gray-500">Version: {model.version} • Created: {new Date(model.createdAt).toLocaleDateString()}</p>
                                        </div>
                                    </div>
                                    <div className="flex flex-col items-end">
                                        <span className="text-sm font-medium text-gray-400">Accuracy</span>
                                        <span className="text-xl font-bold text-green-600">{(model.accuracy * 100).toFixed(1)}%</span>
                                    </div>
                                </div>
                                {model.dataset && (
                                    <div className="mt-4 pt-4 border-t border-gray-50 text-xs text-gray-400">
                                        Trained on: {model.dataset.name}
                                    </div>
                                )}
                            </div>
                        ))
                    )}
                </div>

                {/* API Snippet Panel */}
                <div className="lg:col-span-1">
                    <div className="bg-slate-900 text-white p-6 rounded-2xl shadow-xl sticky top-6">
                        <h3 className="font-semibold mb-4 flex items-center gap-2">
                            <Code size={20} className="text-indigo-400" />
                            API Integration
                        </h3>

                        {!selectedModel ? (
                            <p className="text-slate-400 text-sm">Select a model to view integration code.</p>
                        ) : !snippets ? (
                            <p className="text-slate-400 text-sm animate-pulse">Generating snippets...</p>
                        ) : (
                            <div className="space-y-6">
                                <div>
                                    <div className="flex justify-between items-center mb-2">
                                        <label className="text-xs font-medium text-slate-400">Python</label>
                                        <button
                                            onClick={() => copyToClipboard(snippets.python, 'python')}
                                            className="text-xs flex items-center gap-1 text-indigo-400 hover:text-indigo-300"
                                        >
                                            {copied === 'python' ? <Check size={14} /> : <Copy size={14} />}
                                            {copied === 'python' ? 'Copied' : 'Copy'}
                                        </button>
                                    </div>
                                    <pre className="bg-slate-950 p-3 rounded-lg overflow-x-auto text-xs font-mono text-slate-300 border border-slate-800">
                                        {snippets.python}
                                    </pre>
                                </div>

                                <div>
                                    <div className="flex justify-between items-center mb-2">
                                        <label className="text-xs font-medium text-slate-400">cURL</label>
                                        <button
                                            onClick={() => copyToClipboard(snippets.curl, 'curl')}
                                            className="text-xs flex items-center gap-1 text-indigo-400 hover:text-indigo-300"
                                        >
                                            {copied === 'curl' ? <Check size={14} /> : <Copy size={14} />}
                                            {copied === 'curl' ? 'Copied' : 'Copy'}
                                        </button>
                                    </div>
                                    <pre className="bg-slate-950 p-3 rounded-lg overflow-x-auto text-xs font-mono text-slate-300 border border-slate-800">
                                        {snippets.curl}
                                    </pre>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
