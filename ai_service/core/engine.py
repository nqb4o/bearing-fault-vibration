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
from ydata_profiling import ProfileReport

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
        self.current_model_path = None
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

    async def validate_and_profile(self, dataset_path):
        """
        Validates the dataset schema and generates a ydata-profiling report.
        Handles both single CSV files and directories (recurisvely searching for labeled data).
        Returns: { "valid": bool, "checks": list, "report_path": str }
        """
        checks = []
        is_valid = True
        df_for_profile = None

        if os.path.isdir(dataset_path):
            # 1. Directory Structure & File Discovery
            csv_files = []
            class_counts = {"normal": 0, "inner": 0, "outer": 0}
            
            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    if file.endswith(".csv"):
                        path = os.path.join(root, file)
                        csv_files.append(path)
                        
                        # Label Detection Logic
                        # Strategy 1: Folder Name (Kaggle Structure)
                        path_parts = os.path.normpath(path).split(os.sep)
                        lower_parts = [p.lower() for p in path_parts]
                        
                        # Strategy 2: Filename (Upload Structure: XYZ_IR, XYZ_OR, XYZ_N)
                        lower_name = file.lower()

                        if "normal" in lower_parts or "_n" in lower_name or "normal" in lower_name:
                            class_counts["normal"] += 1
                        elif any(x in lower_parts for x in ["inner race fault", "inner"]) or "_ir" in lower_name:
                            class_counts["inner"] += 1
                        elif any(x in lower_parts for x in ["outer race fault", "outer"]) or "_or" in lower_name:
                            class_counts["outer"] += 1
            
            total_files = len(csv_files)
            if total_files == 0:
                 checks.append({"name": "File Discovery", "status": "Fail", "detail": "No CSV files found in directory."})
                 is_valid = False
            else:
                 checks.append({"name": "File Discovery", "status": "Pass", "detail": f"Found {total_files} CSV files."})
                 
            # Check Class Balance
            missing = []
            if class_counts["normal"] == 0: missing.append("Normal")
            if class_counts["inner"] == 0: missing.append("Inner Race Fault")
            if class_counts["outer"] == 0: missing.append("Outer Race Fault")

            if missing:
                checks.append({
                    "name": "Class Distribution", 
                    "status": "Warning", 
                    "detail": f"Missing classes: {', '.join(missing)}. Ensure folders are named 'Normal', 'Inner Race Fault', 'Outer Race Fault' OR filenames contain '_N', '_IR', '_OR'."
                })
            else:
                checks.append({
                    "name": "Class Distribution", 
                    "status": "Pass", 
                    "detail": f"Found all classes. (N: {class_counts['normal']}, I: {class_counts['inner']}, O: {class_counts['outer']})"
                })

            # SMART SAMPLING for Profiling
            # Instead of first 3 files, take a balanced sample of up to 1000 rows TOTAL from across the classes
            try:
                samples = []
                samples_per_class = 5 # Take 5 files from each class
                
                # Helper to find files for a class
                def get_files_for_class(cls_key):
                    matches = []
                    for f in csv_files:
                        path_parts = os.path.normpath(f).split(os.sep)
                        lower_parts = [p.lower() for p in path_parts]
                        lower_name = os.path.basename(f).lower()
                        
                        if cls_key == "normal" and ("normal" in lower_parts or "_n" in lower_name or "normal" in lower_name):
                             matches.append(f)
                        elif cls_key == "inner" and (any(x in lower_parts for x in ["inner race fault", "inner"]) or "_ir" in lower_name):
                             matches.append(f)
                        elif cls_key == "outer" and (any(x in lower_parts for x in ["outer race fault", "outer"]) or "_or" in lower_name):
                             matches.append(f)
                    return matches

                target_files = []
                target_files.extend(get_files_for_class("normal")[:samples_per_class])
                target_files.extend(get_files_for_class("inner")[:samples_per_class])
                target_files.extend(get_files_for_class("outer")[:samples_per_class])
                
                # If nothing found specific, just take random 10
                if not target_files:
                    target_files = csv_files[:15]

                for p in target_files:
                    # Read only first 100 rows per file to save memory/time
                    df_chunk = pd.read_csv(p, nrows=100) 
                    # If this is 3-column data (Time, Acc1, Acc2) or similar
                    # We might want to rename columns for consistency if they lack headers? 
                    # Kaggle dataset often has NO header, just 3 cols.
                    if df_chunk.shape[1] == 3 and not any(c.lower() in ['time', 'axial', 'radial', 'tangential'] for c in df_chunk.columns):
                         # Assume 3-axis accelerometer data without headers
                         df_chunk.columns = ['Sensor_1', 'Sensor_2', 'Sensor_3']
                    
                    samples.append(df_chunk)
                
                if samples:
                    df_for_profile = pd.concat(samples, ignore_index=True)
                    checks.append({"name": "Data Sampling", "status": "Pass", "detail": f"Created profile using {len(target_files)} representative files (sampled)."})
                else:
                    checks.append({"name": "Data Sampling", "status": "Fail", "detail": "Could not create data sample."})
                    is_valid = False

            except Exception as e:
                checks.append({"name": "Data Reading", "status": "Fail", "detail": f"Failed to read sample files: {str(e)}"})
                is_valid = False

        else:
            # Single File Mode - Legacy Support
            if not os.path.exists(dataset_path):
                 checks.append({"name": "File Check", "status": "Fail", "detail": "File not found."})
                 return {"valid": False, "checks": checks, "report_path": None}
            
            try:
                df_for_profile = pd.read_csv(dataset_path)
                checks.append({"name": "File Access", "status": "Pass", "detail": "Successfully read CSV file."})
            except Exception as e:
                checks.append({"name": "File Access", "status": "Fail", "detail": f"Corrupt or invalid CSV: {str(e)}"})
                is_valid = False


        # 2. Schema Validation (on the loaded dataframe)
        if df_for_profile is not None and is_valid:
            # Column Check
            numeric_cols = df_for_profile.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 0:
                checks.append({"name": "Column Schema", "status": "Pass", "detail": f"Found {len(numeric_cols)} numeric columns."})
            else:
                checks.append({"name": "Column Schema", "status": "Fail", "detail": "No numeric columns found."})
                is_valid = False

            # Missing Values
            if df_for_profile.isnull().values.any():
                missing_count = df_for_profile.isnull().sum().sum()
                checks.append({"name": "Data Integrity", "status": "Warning", "detail": f"Found {missing_count} missing values in sample."})
            else:
                checks.append({"name": "Data Integrity", "status": "Pass", "detail": "No missing values in sample."})

            # Data Length (Check combined length of sample)
            if len(df_for_profile) < 100:
                 checks.append({"name": "Sample Size", "status": "Warning", "detail": f"Sample size {len(df_for_profile)} is small."})
            else:
                 checks.append({"name": "Sample Size", "status": "Pass", "detail": f"Sample size {len(df_for_profile)} OK."})

            # Generate Report
            try:
                report_dir = os.path.join(os.getcwd(), "reports")
                os.makedirs(report_dir, exist_ok=True)
                report_filename = f"profile_{os.path.basename(dataset_path).replace('.csv', '').replace(' ', '_')}.html"
                report_path = os.path.join(report_dir, report_filename)
                
                profile = ProfileReport(df_for_profile, title=f"Dataset Profile: {os.path.basename(dataset_path)}", minimal=True)
                await asyncio.to_thread(profile.to_file, report_path)
                
                return {
                    "valid": is_valid,
                    "checks": checks,
                    "report_path": f"/reports/{report_filename}"
                }
            except Exception as e:
                checks.append({"name": "Profiling", "status": "Fail", "detail": f"Report generation failed: {str(e)}"})
                return {"valid": is_valid, "checks": checks, "report_path": None}
        
        return {
            "valid": is_valid,
            "checks": checks,
            "report_path": None
        }

    def load_model_dynamic(self, model_path_prefix):
        """Loads a specific model version (CNN + Scaler + Features)."""
        try:
            model_path = f"{model_path_prefix}.h5"
            scaler_path = f"{model_path_prefix}_scaler.joblib"
            feature_model_path = f"{model_path_prefix}_features.joblib"
            
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
            
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                
            if os.path.exists(feature_model_path):
                self.feature_model = joblib.load(feature_model_path)
            
            self.current_model_path = model_path_prefix
            return True
        except Exception as e:
            print(f"Failed to load dynamic model: {e}")
            return False

    def download_dataset(self):
        """Downloads dataset from Kaggle."""
        return kagglehub.dataset_download("sumairaziz/subf-v1-0-dataset-bearing-fault-vibration-data")

    def load_and_label_data(self, data_dir):
        filepaths = []
        labels = []

        if os.path.isfile(data_dir):
            # Single file case
            files = [os.path.basename(data_dir)]
            root = os.path.dirname(data_dir)
            iteration = [(root, [], files)]
        else:
            # Directory case
            iteration = os.walk(data_dir)

        for root, dirs, files in iteration:
            for file in files:
                if file.endswith(".csv"):
                    full_path = os.path.join(root, file)
                    
                    # Standardized Labeling Logic
                    path_parts = os.path.normpath(full_path).split(os.sep)
                    lower_parts = [p.lower() for p in path_parts]
                    lower_name = file.lower()
                    
                    label = None
                    
                    # 0: Normal
                    # 1: Inner Race
                    # 2: Outer Race
                    
                    # Check Folder Names first (Kaggle Structure)
                    if "normal" in lower_parts:
                        label = 0
                    elif any(x in lower_parts for x in ["inner race fault", "inner"]):
                        label = 1
                    elif any(x in lower_parts for x in ["outer race fault", "outer"]):
                        label = 2
                    
                    # Fallback/Override: Check Filenames (Upload Structure)
                    # This supports flat directory uploads with named files like XYZ_IR.csv
                    if label is None or True: # Check filename always to catch Mixed cases? No, let's prioritize.
                         # Actually, if we are in a flat dir, folder name might be 'uploads' or 'dataset_123' which is useless.
                         # So if label is NOT found via specific folder names, check file.
                         if label is None:
                             if "_n" in lower_name or "normal" in lower_name:
                                 label = 0
                             elif "_ir" in lower_name or "inner" in lower_name:
                                 label = 1
                             elif "_or" in lower_name or "outer" in lower_name:
                                 label = 2

                    if label is not None:
                        filepaths.append(full_path)
                        labels.append(label)
                    else:
                        # print(f"Skipping unlabeled file: {file} (Path: {full_path})")
                        continue
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

    async def train_model(self, epochs=10, batch_size=32, progress_callback=None, dataset_path=None, model_name="default"):
        """
        Full training pipeline: Download -> Process -> Train CNN + Feature Model -> Save.
        """
        print("Starting training pipeline...")
        
        if dataset_path:
            # If single file provided (for simplicity in this prototype), we might need logic to split or use it.
            # However, existing logic expects a directory with specific naming (inner, outer, normal).
            # If dataset_path is a CSV, we might need to assume it's a merged dataset with labels or just one file.
            # FOR NOW: If dataset_path is a file, we assume it's a CSV with a 'label' column? 
            # OR we assume dataset_path is a DIRECTORY uploaded by admin.
            # Let's assume dataset_path is a single CSV for simplicity of the upload tool, 
            # but for real training we need labeled data. 
            # Let's assume the upload was a ZIP extracted or a single CSV with 'label' column.
            
            if os.path.isfile(dataset_path):
                 # Handle single CSV training (Experimental)
                 print(f"Training on single file: {dataset_path}")
                 df = pd.read_csv(dataset_path)
                 # Expect 'signal' and 'label' columns, or logic to infer.
                 # This is a placeholder for custom dataset logic.
                 # Fallback to download if simple validation.
                 data_path = dataset_path
            else:
                 data_path = dataset_path # It is a directory

        else:
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
        # Ensure we don't sample more than available
        n_background = min(50, X_train.shape[0])
        if n_background > 0:
            self.background_data = X_train[np.random.choice(X_train.shape[0], n_background, replace=False)]
        else:
             self.background_data = X_train # Should not happen given check above but safe fallback

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
        
        # Save history to instance variable
        clean_history = {}
        for k, v in history.history.items():
            clean_history[k] = [float(x) for x in v]
        self.history = clean_history

        # Save to custom path if model_name provided
        base_path = os.path.join(os.path.dirname(MODEL_PATH), model_name)
        
        # Create dir if needed
        os.makedirs(os.path.dirname(base_path), exist_ok=True)

        self.model.save(f"{base_path}.h5")
        joblib.dump(self.scaler, f"{base_path}_scaler.joblib")
        if self.background_data is not None:
             joblib.dump(self.background_data, f"{base_path}_shap.joblib")
        if self.feature_model is not None:
            joblib.dump(self.feature_model, f"{base_path}_features.joblib")
        if self.history is not None:
             joblib.dump(self.history, f"{base_path}_history.joblib")
        
        # Also update default for immediate use
        self.save_resources()
        
        # Evaluate in thread or sync (fast enough)
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
        
        return {
            "status": "Training Complete", 
            "test_acc": float(test_acc),
            "feature_model_acc": float(feat_acc),
            "history": clean_history,
            "model_path": base_path
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
