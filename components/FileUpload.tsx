import React, { useRef, useState } from 'react';
import { UploadCloud, FileText, Loader2, CheckCircle } from 'lucide-react';
import { AppState } from '../types';

interface FileUploadProps {
  onUpload: (file: File) => void;
  appState: AppState;
}

const FileUpload: React.FC<FileUploadProps> = ({ onUpload, appState }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onUpload(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      onUpload(e.target.files[0]);
    }
  };

  const triggerSelect = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="w-full">
      <input 
        ref={fileInputRef}
        type="file" 
        className="hidden" 
        accept=".csv" 
        onChange={handleChange}
      />
      
      <div 
        className={`relative flex flex-col items-center justify-center w-full h-48 rounded-2xl border-2 border-dashed transition-all duration-300 ease-in-out cursor-pointer overflow-hidden
          ${dragActive ? 'border-google-blue bg-blue-50' : 'border-gray-300 bg-white hover:bg-gray-50'}
          ${appState === AppState.PROCESSING ? 'pointer-events-none opacity-80' : ''}
        `}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={triggerSelect}
      >
        {appState === AppState.IDLE || appState === AppState.COMPLETE ? (
           <>
            <div className="p-4 rounded-full bg-blue-50 mb-3 text-google-blue">
              <UploadCloud size={24} />
            </div>
            <p className="text-sm font-medium text-google-text">
              <span className="text-google-blue hover:underline">Click to upload</span> or drag and drop
            </p>
            <p className="text-xs text-gray-400 mt-1">CSV files only (min 1024 rows)</p>
           </>
        ) : appState === AppState.PROCESSING ? (
          <>
             <div className="absolute inset-0 bg-white/80 z-10 flex flex-col items-center justify-center">
               <Loader2 className="animate-spin text-google-blue mb-2" size={32} />
               <p className="text-sm font-medium text-google-text">Analyzing Signal...</p>
               <p className="text-xs text-gray-500 mt-1">Calculating SHAP Values & RMS</p>
             </div>
          </>
        ) : null}
        
        {/* Success overlay state if needed, though usually handled by parent changing view */}
      </div>
    </div>
  );
};

export default FileUpload;
