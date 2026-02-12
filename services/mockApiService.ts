import { AnalysisResult, DiagnosisStatus, TrainingLog, TrainingConfig } from '../types';
import { MOCK_SIGNAL, REFERENCE_DATA } from '../constants';

// --- MOCK CONSTANTS FOR SIMULATION ---
const SAMPLE_COUNT = 1000;

export const uploadAndAnalyze = async (file: File): Promise<AnalysisResult> => {
  return new Promise((resolve) => {
    // Simulate Processing Delay (SHAP GradientExplainer takes time)
    setTimeout(() => {
      const rand = Math.random();
      let status = DiagnosisStatus.NORMAL;
      let rms = 0.8;
      let shapImpact = 0.05;
      
      // Logic mirroring the Python probabilistic generation
      if (rand > 0.66) {
        status = DiagnosisStatus.INNER_RACE_FAULT;
        rms = 2.4 + (Math.random() * 0.5);
        shapImpact = 0.75 + (Math.random() * 0.2);
      } else if (rand > 0.33) {
        status = DiagnosisStatus.OUTER_RACE_FAULT;
        rms = 1.6 + (Math.random() * 0.4);
        shapImpact = 0.45 + (Math.random() * 0.2);
      } else {
        rms = 0.5 + (Math.random() * 0.3);
        shapImpact = 0.02 + (Math.random() * 0.05);
      }

      // Generate Probabilities based on Status
      let probs = [0.05, 0.05, 0.05]; // Base noise
      if (status === DiagnosisStatus.NORMAL) {
        probs[0] = 0.8 + (Math.random() * 0.15);
        probs[1] = (1 - probs[0]) * 0.3;
        probs[2] = (1 - probs[0]) * 0.7;
      } else if (status === DiagnosisStatus.INNER_RACE_FAULT) {
        probs[1] = 0.8 + (Math.random() * 0.15);
        probs[0] = (1 - probs[1]) * 0.2;
        probs[2] = (1 - probs[1]) * 0.8;
      } else {
        probs[2] = 0.8 + (Math.random() * 0.15);
        probs[0] = (1 - probs[2]) * 0.4;
        probs[1] = (1 - probs[2]) * 0.6;
      }

      // Normalize
      const total = probs.reduce((a, b) => a + b, 0);
      const normalizedProbs = probs.map(p => p / total);
      const confidence = Math.max(...normalizedProbs);

      resolve({
        id: Math.random().toString(36).substr(2, 9),
        timestamp: new Date().toISOString(),
        status: status,
        confidence: confidence,
        rms: rms,
        shapImpact: shapImpact,
        kurtosis: 3.5 + (Math.random() * 2), // Mock kurtosis
        signalSnippet: MOCK_SIGNAL.map(v => v * (rms * 0.6)), // Scale signal visualization
        probabilities: [
            { label: DiagnosisStatus.NORMAL, value: normalizedProbs[0] },
            { label: DiagnosisStatus.INNER_RACE_FAULT, value: normalizedProbs[1] },
            { label: DiagnosisStatus.OUTER_RACE_FAULT, value: normalizedProbs[2] }
        ]
      });
    }, 2000);
  });
};

// --- TRAINING SIMULATION (Admin) ---

export const triggerDatasetDownload = async (): Promise<{ samples: number, status: string }> => {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({ samples: 4800, status: 'Ready' });
    }, 2500); // Simulate Kaggle download time
  });
};

export const startModelTraining = async (
  config: TrainingConfig, 
  onLog: (log: TrainingLog) => void
): Promise<void> => {
  let currentEpoch = 0;
  
  // Initial Mock Loss
  let loss = 0.8;
  let accuracy = 0.45;
  let val_loss = 0.9;
  let val_accuracy = 0.40;

  return new Promise((resolve) => {
    const interval = setInterval(() => {
      currentEpoch++;
      
      // Simulate convergence
      loss = Math.max(0.1, loss * 0.9 + (Math.random() * 0.05 - 0.025));
      accuracy = Math.min(0.99, accuracy * 1.05 + (Math.random() * 0.02));
      val_loss = Math.max(0.15, val_loss * 0.92 + (Math.random() * 0.05 - 0.025));
      val_accuracy = Math.min(0.96, val_accuracy * 1.04 + (Math.random() * 0.02));

      onLog({
        epoch: currentEpoch,
        loss,
        accuracy,
        val_loss,
        val_accuracy
      });

      if (currentEpoch >= config.epochs) {
        clearInterval(interval);
        resolve();
      }
    }, 800); // 800ms per epoch
  });
};

export const getTrainingStats = async () => {
  return REFERENCE_DATA;
};