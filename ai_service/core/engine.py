import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import kurtosis
import shap
import joblib
import kagglehub
import asyncio
from .config import WINDOW_SIZE, STRIDE, TEST_SIZE, RANDOM_SEED, MODEL_PATH, SCALER_PATH, SHAP_BACKGROUND_PATH, FEATURE_MODEL_PATH, HISTORY_PATH
from .features import extract_features_batch, FEATURE_NAMES, FEATURE_CATEGORIES

# Global State (Simulating in-memory persistence for now, but backed by files)
class AIEngine:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.X_train = None
        self.y_train = None
        self.label_map = {0: "Normal", 1: "Inner Race Fault", 2: "Outer Race Fault"}
        self.background_data = None
        self.feature_model = None
        self.history = None
        self.load_resources()

    def load_resources(self):
        """Loads model and scaler from disk if available."""
        if os.path.exists(MODEL_PATH):
            try:
                self.model = tf.keras.models.load_model(MODEL_PATH)
                print(f"Loaded model from {MODEL_PATH}")
            except Exception as e:
                print(f"Failed to load model: {e}")

        if os.path.exists(SCALER_PATH):
            try:
                self.scaler = joblib.load(SCALER_PATH)
                print(f"Loaded scaler from {SCALER_PATH}")
            except Exception as e:
                print(f"Failed to load scaler: {e}")

        if os.path.exists(SHAP_BACKGROUND_PATH):
            try:
                self.background_data = joblib.load(SHAP_BACKGROUND_PATH)
                print(f"Loaded SHAP background data from {SHAP_BACKGROUND_PATH}")
            except Exception as e:
                print(f"Failed to load SHAP background data: {e}")

        if os.path.exists(FEATURE_MODEL_PATH):
            try:
                self.feature_model = joblib.load(FEATURE_MODEL_PATH)
                print(f"Loaded feature model from {FEATURE_MODEL_PATH} (type: {type(self.feature_model).__name__})")
            except Exception as e:
                print(f"Failed to load feature model: {e} — will retrain to fix")
                self.feature_model = None
        else:
            print(f"Feature model not found at {FEATURE_MODEL_PATH} — train to create it")

        if os.path.exists(HISTORY_PATH):
            try:
                self.history = joblib.load(HISTORY_PATH)
                print(f"Loaded training history from {HISTORY_PATH}")
            except Exception as e:
                print(f"Failed to load history: {e}")

    def save_resources(self):
        """Saves model and scaler to disk."""
        if self.model:
            self.model.save(MODEL_PATH)
        if self.scaler:
            joblib.dump(self.scaler, SCALER_PATH)
        if self.background_data is not None:
             joblib.dump(self.background_data, SHAP_BACKGROUND_PATH)
        if self.feature_model is not None:
            joblib.dump(self.feature_model, FEATURE_MODEL_PATH)
        if self.history is not None:
             joblib.dump(self.history, HISTORY_PATH)

    def download_dataset(self):
        """Downloads dataset from Kaggle."""
        return kagglehub.dataset_download("sumairaziz/subf-v1-0-dataset-bearing-fault-vibration-data")

    def load_and_label_data(self, data_dir):
        filepaths = []
        labels = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".csv"):
                    full_path = os.path.join(root, file)
                    lower_name = (file + root).lower()
                    if "normal" in lower_name:
                        label = 0
                    elif "inner" in lower_name:
                        label = 1
                    elif "outer" in lower_name:
                        label = 2
                    else:
                        continue
                    filepaths.append(full_path)
                    labels.append(label)
        return filepaths, labels

    def create_sliding_windows(self, signal, window_size, stride):
        if len(signal) < window_size:
            return np.array([])
        num_windows = (len(signal) - window_size) // stride + 1
        if num_windows <= 0:
            return np.array([])
        shp = (num_windows, window_size)
        strides = (signal.strides[0] * stride, signal.strides[0])
        return np.lib.stride_tricks.as_strided(signal, shape=shp, strides=strides)

    async def train_model(self, epochs=10, batch_size=32, progress_callback=None):
        """
        Full training pipeline: Download -> Process -> Train CNN + Feature Model -> Save.
        """
        print("Starting training pipeline...")
        # Download and Data Prep (can stay sync or be moved to thread if heavy)
        data_path = self.download_dataset()
        filepaths, labels = self.load_and_label_data(data_path)

        all_X = []
        all_y = []

        # Process files
        for i, (fp, lbl) in enumerate(zip(filepaths, labels)):
            try:
                df = pd.read_csv(fp)
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    signal = df[numeric_cols[0]].values
                    windows = self.create_sliding_windows(signal, WINDOW_SIZE, STRIDE)
                    if len(windows) > 0:
                        all_X.append(windows)
                        all_y.extend([lbl] * len(windows))
            except Exception:
                pass

        if not all_X:
            raise ValueError("No valid data found for training.")

        X = np.concatenate(all_X, axis=0)
        y = np.array(all_y)
        
        # Keep raw windows for feature extraction (before scaling)
        X_raw = X.copy()  # shape: (N, W)
        
        X = np.expand_dims(X, axis=-1)

        # Scaling (for CNN)
        self.scaler = StandardScaler()
        N, W, C = X.shape
        X_reshaped = X.reshape(-1, C)
        X_scaled = self.scaler.fit_transform(X_reshaped)
        self.X_train_full = X_scaled.reshape(N, W, C)
        self.y_train_full = y

    
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_train_full, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
        )
        
        # Split raw windows with the SAME random state to keep alignment
        X_raw_train, X_raw_test, _, _ = train_test_split(
            X_raw, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
        )
        
        # Select background data for SHAP (subset of training data)
        self.background_data = X_train[np.random.choice(X_train.shape[0], 50, replace=False)]

        # ───────────────────────────────────────────────────────────────────
        # Train Secondary Feature-Based Model for SHAP Explainability
        # Uses RAW (unscaled) windows so features are physically meaningful
        # ───────────────────────────────────────────────────────────────────
        print("Extracting vibration features for secondary model...")
        X_feat_train = extract_features_batch(X_raw_train)
        X_feat_test = extract_features_batch(X_raw_test)

        # Replace any NaN/inf with 0
        X_feat_train = np.nan_to_num(X_feat_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_feat_test = np.nan_to_num(X_feat_test, nan=0.0, posinf=0.0, neginf=0.0)

        print(f"Training feature model on {X_feat_train.shape[0]} samples with {X_feat_train.shape[1]} features...")
        self.feature_model = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1
        )
        self.feature_model.fit(X_feat_train, y_train)
        feat_acc = self.feature_model.score(X_feat_test, y_test)
        print(f"Feature model accuracy: {feat_acc:.4f}")

        
        # Build CNN Model
        num_classes = len(self.label_map)
        self.model = models.Sequential([
            layers.Input(shape=(W, C)),
            layers.Conv1D(32, 3, activation='relu', padding='same'),
            layers.MaxPooling1D(2),
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.GlobalAveragePooling1D(),
            layers.Dense(64, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Custom Callback for Progress
        loop = asyncio.get_running_loop()

        class ProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if progress_callback:
                    metrics = {
                        "epoch": epoch + 1,
                        "loss": logs.get('loss'),
                        "accuracy": logs.get('accuracy'),
                        "val_loss": logs.get('val_loss'),
                        "val_accuracy": logs.get('val_accuracy')
                    }
                    asyncio.run_coroutine_threadsafe(progress_callback(metrics), loop)

        # Run model.fit in a separate thread so it doesn't block the main event loop
        def run_fit():
            return self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[ProgressCallback()],
                verbose=0
            )

        history = await loop.run_in_executor(None, run_fit)
        self.history = history.history

        self.save_resources()
        
        # Evaluate in thread or sync (fast enough)
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
        
        return {
            "status": "Training Complete", 
            "test_acc": test_acc,
            "feature_model_acc": feat_acc,
            "history": history.history
        }

    def predict(self, signal_values):
        """
        Inference logic: Preprocess -> Scale -> Predict -> SHAP Explain with named features.
        """
        if not self.model or not self.scaler:
            raise ValueError("Model or Scaler not loaded.")

        windows = self.create_sliding_windows(signal_values, WINDOW_SIZE, STRIDE)
        if len(windows) == 0:
            raise ValueError(f"Signal too short. Need {WINDOW_SIZE} samples.")

        # Scale
        windows_raw = windows.copy()  # Keep unscaled for feature extraction
        windows = np.expand_dims(windows, axis=-1)
        N, W, C = windows.shape
        windows_reshaped = windows.reshape(-1, C)
        windows_scaled = self.scaler.transform(windows_reshaped)
        X_new = windows_scaled.reshape(N, W, C)

        # Predict with CNN
        probs = self.model.predict(X_new)
        pred_indices = np.argmax(probs, axis=1)
        
        # Aggregate
        final_idx = int(np.bincount(pred_indices).argmax())
        final_label = self.label_map[final_idx]
        confidence = float(np.mean(probs, axis=0)[final_idx])

        # ───────────────────────────────────────────────────────────────────
        # SHAP Explanation with Named Vibration Features
        # ───────────────────────────────────────────────────────────────────
        shap_explanation = None
        try:
            if self.feature_model is not None:
                # Extract features from all windows
                X_feat = extract_features_batch(windows_raw)
                X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0)

                # Compute SHAP values using TreeExplainer (fast & exact)
                explainer = shap.TreeExplainer(self.feature_model)
                shap_values = explainer.shap_values(X_feat)

                # shap_values is a list of arrays [class_0, class_1, class_2]
                # Each array has shape (N_windows, N_features)
                # We want SHAP values for the predicted class, averaged across windows
                if isinstance(shap_values, list):
                    # Old SHAP API returns a list of [ (N, 17), (N, 17), (N, 17) ]
                    shap_for_class = shap_values[final_idx]
                elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                    # New SHAP API returns (N, 17, 3)
                    shap_for_class = shap_values[:, :, final_idx]
                else:
                    # Binary or single-output fallback
                    shap_for_class = shap_values

                # Average across windows
                mean_shap = np.mean(shap_for_class, axis=0)  # Should be (17,)
                mean_features = np.mean(X_feat, axis=0)  # (17,)

                # Get base value (expected value)
                ev = explainer.expected_value
                if isinstance(ev, (list, np.ndarray)):
                    base_value = float(ev[final_idx])
                else:
                    base_value = float(ev)

                shap_explanation = {
                    "features": FEATURE_NAMES,
                    "categories": [FEATURE_CATEGORIES[f] for f in FEATURE_NAMES],
                    "values": [float(v) for v in mean_features],
                    "shap_values": [float(v) for v in mean_shap],
                    "base_value": base_value,
                    "predicted_class_index": final_idx,
                    "predicted_class": final_label,
                }
        except Exception as e:
            print(f"SHAP Feature Explanation Error: {e}")
            import traceback
            traceback.print_exc()

        return {
            "label": final_label,
            "confidence": confidence,
            "predictions": probs.tolist(),
            "shap_explanation": shap_explanation,
        }

engine = AIEngine()
