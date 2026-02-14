'use client';

import { useState, useEffect } from 'react';
import { Upload, FileText, Activity, CheckCircle, AlertCircle, RefreshCw } from 'lucide-react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:4000/api';

export default function DatasetsPage() {
    const [datasets, setDatasets] = useState<any[]>([]);
    const [uploading, setUploading] = useState(false);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');
    const [selectedFile, setSelectedFile] = useState<File | null>(null);

    useEffect(() => {
        fetchDatasets();
    }, []);

    const fetchDatasets = async () => {
        try {
            setLoading(true);
            const token = localStorage.getItem('token');
            const response = await fetch(`${API_URL}/admin/datasets`, {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });
            if (response.ok) {
                const data = await response.json();
                setDatasets(data);
            } else {
                setError('Failed to fetch datasets');
            }
        } catch (err) {
            setError('Error connecting to server');
        } finally {
            setLoading(false);
        }
    };

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            setSelectedFile(e.target.files[0]);
        }
    };

    const handleUpload = async () => {
        if (!selectedFile) return;

        setUploading(true);
        setError('');

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const token = localStorage.getItem('token');
            const response = await fetch(`${API_URL}/admin/datasets/upload`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`
                },
                body: formData
            });

            if (response.ok) {
                setSelectedFile(null);
                fetchDatasets();
            } else {
                const data = await response.json();
                setError(data.error || 'Upload failed');
            }
        } catch (err) {
            setError('Upload error');
        } finally {
            setUploading(false);
        }
    };

    const triggerValidation = async (id: number) => {
        try {
            const token = localStorage.getItem('token');
            const response = await fetch(`${API_URL}/admin/datasets/${id}/validate`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });
            if (response.ok) {
                fetchDatasets(); // Refresh to show 'processing' status
            }
        } catch (err) {
            console.error(err);
        }
    };

    return (
        <div className="space-y-6">
            <h1 className="text-2xl font-bold text-gray-800">Dataset Management</h1>

            {/* Upload Section */}
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100">
                <h2 className="text-lg font-semibold mb-4">Upload New Dataset</h2>
                <div className="flex items-center gap-4">
                    <input
                        type="file"
                        accept=".csv"
                        onChange={handleFileChange}
                        className="block w-full text-sm text-gray-500
                            file:mr-4 file:py-2 file:px-4
                            file:rounded-full file:border-0
                            file:text-sm file:font-semibold
                            file:bg-indigo-50 file:text-indigo-700
                            hover:file:bg-indigo-100"
                    />
                    <button
                        onClick={handleUpload}
                        disabled={!selectedFile || uploading}
                        className={`px-6 py-2 rounded-full font-semibold text-white transition flex items-center gap-2
                            ${!selectedFile || uploading ? 'bg-gray-400 cursor-not-allowed' : 'bg-indigo-600 hover:bg-indigo-700'}
                        `}
                    >
                        {uploading ? 'Uploading...' : 'Upload'}
                        <Upload size={18} />
                    </button>
                </div>
                {error && <p className="text-red-500 mt-2 text-sm">{error}</p>}
            </div>

            {/* Datasets List */}
            <div className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
                <div className="p-6 border-b border-gray-100 flex justify-between items-center">
                    <h2 className="text-lg font-semibold">Uploaded Datasets</h2>
                    <button onClick={fetchDatasets} className="text-gray-500 hover:text-indigo-600">
                        <RefreshCw size={20} />
                    </button>
                </div>

                {loading ? (
                    <div className="p-6 text-center text-gray-500">Loading datasets...</div>
                ) : datasets.length === 0 ? (
                    <div className="p-6 text-center text-gray-500">No datasets uploaded yet.</div>
                ) : (
                    <table className="w-full text-left">
                        <thead className="bg-gray-50 text-gray-600 text-sm font-medium">
                            <tr>
                                <th className="px-6 py-4">Name</th>
                                <th className="px-6 py-4">Status</th>
                                <th className="px-6 py-4">Size</th>
                                <th className="px-6 py-4">Rows</th>
                                <th className="px-6 py-4">Actions</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-100">
                            {datasets.map((ds) => (
                                <tr key={ds.id} className="hover:bg-gray-50 transition">
                                    <td className="px-6 py-4 font-medium flex items-center gap-3">
                                        <div className="p-2 bg-indigo-50 text-indigo-600 rounded-lg">
                                            <FileText size={20} />
                                        </div>
                                        {ds.name}
                                    </td>
                                    <td className="px-6 py-4">
                                        <span className={`inline-flex items-center gap-1 px-3 py-1 rounded-full text-xs font-semibold
                                            ${ds.status === 'ready' ? 'bg-green-100 text-green-700' :
                                                ds.status === 'processing' ? 'bg-yellow-100 text-yellow-700' :
                                                    ds.status === 'error' ? 'bg-red-100 text-red-700' :
                                                        'bg-gray-100 text-gray-700'}
                                        `}>
                                            {ds.status === 'ready' && <CheckCircle size={12} />}
                                            {ds.status === 'processing' && <Activity size={12} className="animate-spin" />}
                                            {ds.status === 'error' && <AlertCircle size={12} />}
                                            {ds.status.charAt(0).toUpperCase() + ds.status.slice(1)}
                                        </span>
                                    </td>
                                    <td className="px-6 py-4 text-gray-600">{(ds.size / 1024 / 1024).toFixed(2)} MB</td>
                                    <td className="px-6 py-4 text-gray-600">{ds.rowCount}</td>
                                    <td className="px-6 py-4">
                                        <div className="flex gap-2">
                                            {ds.status === 'uploaded' && (
                                                <button
                                                    onClick={() => triggerValidation(ds.id)}
                                                    className="text-sm text-indigo-600 hover:text-indigo-800 font-medium"
                                                >
                                                    Analyze
                                                </button>
                                            )}
                                            {ds.status === 'ready' && (
                                                <span className="text-sm text-gray-400">Validated</span>
                                            )}
                                        </div>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                )}
            </div>
        </div>
    );
}
