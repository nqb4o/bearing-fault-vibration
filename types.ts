export enum DiagnosisStatus {
  NORMAL = 'Normal',
  INNER_RACE_FAULT = 'Inner Race Fault',
  OUTER_RACE_FAULT = 'Outer Race Fault'
}

export interface AnalysisResult {
  id: string;
  timestamp: string;
  status: DiagnosisStatus;
  confidence: number;
  rms: number;
  shapImpact: number;
  kurtosis: number;
  signalSnippet: number[];
  probabilities: { label: string; value: number }[];
}

export interface FeaturePoint {
  x: number; // SHAP Impact
  y: number; // RMS
  type: 'background' | 'current';
  label: string;
}

export enum AppState {
  IDLE,
  UPLOADING,
  PROCESSING,
  COMPLETE,
  ERROR
}

export enum ViewMode {
  DIAGNOSTICS = 'diagnostics',
  HISTORY = 'history',
  TRAINING = 'training',
  SETTINGS = 'settings'
}

export interface TrainingLog {
  epoch: number;
  loss: number;
  accuracy: number;
  val_loss: number;
  val_accuracy: number;
}

export interface TrainingConfig {
  epochs: number;
  batchSize: number;
}