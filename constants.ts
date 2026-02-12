import { FeaturePoint, DiagnosisStatus } from './types';

// Simulating the "Reference Data" from the Python StandardScaler/Training set
// This allows the Scatter Plot to show the "Manifold" of known states.
export const REFERENCE_DATA: FeaturePoint[] = [
  // Normal Cluster (Low RMS, Low SHAP)
  ...Array.from({ length: 40 }).map(() => ({
    x: Math.random() * 0.1, 
    y: Math.random() * 0.5 + 0.5, 
    type: 'background' as const, 
    label: DiagnosisStatus.NORMAL 
  })),
  // Inner Race Fault Cluster (High RMS, High SHAP)
  ...Array.from({ length: 40 }).map(() => ({
    x: Math.random() * 0.3 + 0.5, 
    y: Math.random() * 1.5 + 1.5, 
    type: 'background' as const, 
    label: DiagnosisStatus.INNER_RACE_FAULT 
  })),
  // Outer Race Fault Cluster (Medium RMS, Medium/High SHAP)
  ...Array.from({ length: 40 }).map(() => ({
    x: Math.random() * 0.2 + 0.3, 
    y: Math.random() * 1.0 + 1.0, 
    type: 'background' as const, 
    label: DiagnosisStatus.OUTER_RACE_FAULT 
  })),
];

export const MOCK_SIGNAL = Array.from({ length: 128 }).map((_, i) => Math.sin(i * 0.2) + Math.random() * 0.5);
