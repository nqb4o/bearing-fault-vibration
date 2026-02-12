import React from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';

interface SignalChartProps {
  data: number[];
}

const SignalChart: React.FC<SignalChartProps> = ({ data }) => {
  const chartData = data.map((val, idx) => ({ index: idx, value: val }));

  return (
    <div className="bg-white p-6 rounded-2xl border border-gray-100 shadow-sm">
      <h3 className="text-google-text font-semibold text-lg mb-1">Vibration Signature</h3>
      <p className="text-google-subtext text-sm mb-4">Raw accelerometer data (Time Domain - 1024 window)</p>
      
      <div className="w-full h-[200px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
            <XAxis hide dataKey="index" />
            <YAxis hide domain={['auto', 'auto']} />
            <Tooltip 
              contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}
              itemStyle={{ color: '#1A73E8', fontSize: '12px' }}
              labelStyle={{ display: 'none' }}
            />
            <Line 
              type="monotone" 
              dataKey="value" 
              stroke="#1A73E8" 
              strokeWidth={2} 
              dot={false} 
              activeDot={{ r: 6 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default SignalChart;
