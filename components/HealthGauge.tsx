import React from 'react';
import { DiagnosisStatus } from '../types';

interface HealthGaugeProps {
  score: number; // 0 to 1
  status: DiagnosisStatus;
}

const HealthGauge: React.FC<HealthGaugeProps> = ({ score, status }) => {
  // Config
  const radius = 80;
  const stroke = 12;
  const normalizedScore = score * 100;
  const circumference = normalizedScore * 2 * Math.PI * radius;
  
  // Colors based on status
  let color = '#34A853'; // Google Green
  if (status === DiagnosisStatus.INNER_RACE_FAULT) color = '#EA4335'; // Google Red
  if (status === DiagnosisStatus.OUTER_RACE_FAULT) color = '#FBBC04'; // Google Yellow

  const strokeDasharray = `${(normalizedScore / 100) * (2 * Math.PI * radius)} ${2 * Math.PI * radius}`;
  const offset = 2 * Math.PI * radius * 0.25; // Start from top

  return (
    <div className="flex flex-col items-center justify-center p-6 bg-white rounded-2xl border border-gray-100 shadow-sm relative overflow-hidden">
      <h3 className="text-google-subtext font-medium text-sm mb-4 uppercase tracking-wider">AI Confidence Score</h3>
      
      <div className="relative w-48 h-48 flex items-center justify-center">
        <svg className="w-full h-full transform -rotate-90" viewBox="0 0 200 200">
          {/* Background Circle */}
          <circle
            cx="100"
            cy="100"
            r={radius}
            fill="none"
            stroke="#F1F3F4"
            strokeWidth={stroke}
          />
          {/* Progress Circle */}
          <circle
            cx="100"
            cy="100"
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth={stroke}
            strokeDasharray={strokeDasharray}
            strokeLinecap="round"
            className="transition-all duration-1000 ease-out"
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-4xl font-bold text-google-text">
            {Math.round(normalizedScore)}%
          </span>
          <span className={`text-xs font-semibold px-2 py-0.5 rounded-full mt-2 ${
            status === DiagnosisStatus.NORMAL ? 'bg-green-50 text-green-700' :
            status === DiagnosisStatus.INNER_RACE_FAULT ? 'bg-red-50 text-red-700' : 'bg-yellow-50 text-yellow-700'
          }`}>
            {status}
          </span>
        </div>
      </div>
      <p className="text-xs text-center text-gray-400 mt-4 max-w-[200px]">
        Based on GradientExplainer SHAP analysis of 1024-point windows.
      </p>
    </div>
  );
};

export default HealthGauge;
