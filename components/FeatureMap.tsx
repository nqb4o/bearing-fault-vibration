import React from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from 'recharts';
import { FeaturePoint, DiagnosisStatus } from '../types';
import { REFERENCE_DATA } from '../constants';

interface FeatureMapProps {
  currentResult: { rms: number; shapImpact: number; status: DiagnosisStatus } | null;
}

const FeatureMap: React.FC<FeatureMapProps> = ({ currentResult }) => {
  
  // Combine reference data with current point if exists
  const data = [...REFERENCE_DATA];
  if (currentResult) {
    data.push({
      x: currentResult.shapImpact,
      y: currentResult.rms,
      type: 'current',
      label: currentResult.status
    });
  }

  // Custom Tooltip
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const dataPoint = payload[0].payload;
      return (
        <div className="bg-white p-3 border border-gray-100 shadow-lg rounded-lg text-xs">
          <p className="font-semibold mb-1">{dataPoint.label}</p>
          <p className="text-gray-500">RMS: {dataPoint.y.toFixed(2)}</p>
          <p className="text-gray-500">SHAP: {dataPoint.x.toFixed(2)}</p>
          {dataPoint.type === 'current' && <p className="text-blue-600 font-bold mt-1">Current Scan</p>}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="bg-white p-6 rounded-2xl border border-gray-100 shadow-sm h-full flex flex-col">
      <div className="flex justify-between items-start mb-6">
        <div>
          <h3 className="text-google-text font-semibold text-lg">AI Feature Map</h3>
          <p className="text-google-subtext text-sm">Signal Energy (RMS) vs. Model Sensitivity (SHAP)</p>
        </div>
        <div className="flex gap-2 text-xs">
          <div className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-gray-300"></span> History</div>
          <div className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-blue-600"></span> You</div>
        </div>
      </div>

      <div className="flex-grow w-full h-[300px]">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis 
              type="number" 
              dataKey="x" 
              name="SHAP Impact" 
              label={{ value: 'AI Impact (SHAP)', position: 'insideBottom', offset: -10, fontSize: 12, fill: '#9AA0A6' }} 
              tick={{ fontSize: 10, fill: '#9AA0A6' }}
              domain={[0, 'auto']}
            />
            <YAxis 
              type="number" 
              dataKey="y" 
              name="RMS" 
              label={{ value: 'Vibration RMS', angle: -90, position: 'insideLeft', fontSize: 12, fill: '#9AA0A6' }} 
              tick={{ fontSize: 10, fill: '#9AA0A6' }}
              domain={[0, 'auto']}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: '3 3' }} />
            <Scatter name="Points" data={data} fill="#8884d8">
              {data.map((entry, index) => {
                if (entry.type === 'current') {
                  return <Cell key={`cell-${index}`} fill="#1A73E8" stroke="#fff" strokeWidth={3} r={8} />; // Big Blue Dot
                }
                // Color background points by their class roughly
                let color = '#DADCE0'; // Gray default
                if (entry.label === DiagnosisStatus.INNER_RACE_FAULT) color = '#FAD2CF'; // Light Red
                if (entry.label === DiagnosisStatus.OUTER_RACE_FAULT) color = '#FCE8B2'; // Light Yellow
                if (entry.label === DiagnosisStatus.NORMAL) color = '#CEEAD6'; // Light Green
                return <Cell key={`cell-${index}`} fill={color} />;
              })}
            </Scatter>
            {/* Safe Zone Reference */}
            <ReferenceLine y={0.8} stroke="#34A853" strokeDasharray="3 3" label={{ value: "Safe Threshold", position: 'insideTopRight', fill: '#34A853', fontSize: 10 }} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default FeatureMap;
