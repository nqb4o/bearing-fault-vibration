import streamlit as st
import kagglehub
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import kurtosis
import seaborn as sns
import tempfile

try:
    import shap
except ImportError:
    st.warning("SHAP library not found. Explainability features will be disabled. Run `pip install shap`.")
    shap = None

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="BearingGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

WINDOW_SIZE = 1024
STRIDE = 512
TEST_SIZE = 0.2
RANDOM_SEED = 42


# -----------------------------------------------------------------------------
# Helper Functions: Data & Model
# -----------------------------------------------------------------------------

def download_dataset():
    """Downloads the dataset using kagglehub and returns the path."""
    with st.spinner("Downloading dataset from Kaggle... (This may take a minute)"):
        path = kagglehub.dataset_download("sumairaziz/subf-v1-0-dataset-bearing-fault-vibration-data")
    return path


def validate_uploaded_file(df):
    """
    Validates the uploaded dataframe for Field Diagnostics.
    Returns (True, message) if valid, (False, error_message) if invalid.
    """
    # 1. Column Check
    candidates = ['vibration_signal', 'amplitude', 'vibration', 'signal']
    found_col = None
    for col in df.columns:
        if col.lower() in candidates:
            found_col = col
            break

    # If not found by name, check if 1st column is numeric (fallback)
    if not found_col:
        first_col = df.columns[0]
        if pd.api.types.is_numeric_dtype(df[first_col]):
            found_col = first_col
        else:
            return False, f"❌ Column check failed. Could not find a numeric signal column. Candidates: {candidates}", None

    # 2. Data Type Check
    if not pd.api.types.is_numeric_dtype(df[found_col]):
        return False, f"❌ Non-numeric data found in column '{found_col}'. Please ensure the file contains only sensor readings.", None

    # 3. Missing Values
    if df[found_col].isnull().any():
        return False, "❌ Empty values detected (NaN). Please clean your data.", None

    # 4. Data Length
    if len(df) < WINDOW_SIZE:
        return False, f"❌ Insufficient data. File has {len(df)} rows, but model requires at least {WINDOW_SIZE}.", None

    return True, "✅ File Validated", df[found_col].values


def load_and_label_data(data_dir):
    filepaths = []
    labels = []
    label_map = {0: "Normal", 1: "Inner Race Fault", 2: "Outer Race Fault"}

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

    return filepaths, labels, label_map


def create_sliding_windows(signal, window_size, stride):
    if len(signal) < window_size:
        return np.array([])
    num_windows = (len(signal) - window_size) // stride + 1
    if num_windows <= 0:
        return np.array([])

    shp = (num_windows, window_size)
    strides = (signal.strides[0] * stride, signal.strides[0])
    windows = np.lib.stride_tricks.as_strided(signal, shape=shp, strides=strides)
    return windows


@st.cache_data
def load_and_process_data(data_dir):
    filepaths, labels, label_map = load_and_label_data(data_dir)
    all_X = []
    all_y = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, (fp, lbl) in enumerate(zip(filepaths, labels)):
        status_text.text(f"Processing {i + 1}/{len(filepaths)}: {os.path.basename(fp)}")
        try:
            df = pd.read_csv(fp)
            # Simple column finder for training data
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                signal = df[numeric_cols[0]].values
                windows = create_sliding_windows(signal, WINDOW_SIZE, STRIDE)
                if len(windows) > 0:
                    all_X.append(windows)
                    all_y.extend([lbl] * len(windows))
        except:
            pass
        progress_bar.progress((i + 1) / len(filepaths))

    status_text.empty()
    progress_bar.empty()

    if not all_X:
        return None, None, None, None, None, None

    X = np.concatenate(all_X, axis=0)
    y = np.array(all_y)
    X = np.expand_dims(X, axis=-1)

    scaler = StandardScaler()
    N, W, C = X.shape
    X_reshaped = X.reshape(-1, C)
    X_scaled = scaler.fit_transform(X_reshaped)
    X = X_scaled.reshape(N, W, C)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)

    return X_train, X_test, y_train, y_test, label_map, scaler


def build_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(32, 3, activation='relu', padding='same'),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 3, activation='relu', padding='same'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def compute_shap_explanation(model, X_train, input_window):
    """
    Computes SHAP values for a single input window using GradientExplainer.
    Uses a summarized background from X_train for performance.
    """
    if shap is None:
        return None, None

    with st.spinner("Analyzing signal patterns... this may take a moment"):
        # Summarize background (100 samples)
        # Handle case where X_train might be smaller than 100
        n_samples = min(100, X_train.shape[0])
        background = X_train[np.random.choice(X_train.shape[0], n_samples, replace=False)]

        # Initialize Explainer
        # tf.compat.v1.disable_v2_behavior() # Sometimes needed for SHAP deep/gradient explainer issues, but try without first
        explainer = shap.GradientExplainer(model, background)

        # Compute SHAP values
        # Input must be (1, 1024, 1)
        input_reshaped = np.expand_dims(input_window, axis=0)
        shap_values = explainer.shap_values(input_reshaped)

        # shap_values is a list of arrays (one for each class)
        # or a single array depending on version/model output.
        # For multi-class classification, it returns a list of [samples, sequence_length, channels] arrays.
        return shap_values, explainer


class StreamlitCallback(tf.keras.callbacks.Callback):
    def __init__(self, plot_placeholder):
        super().__init__()
        self.plot_placeholder = plot_placeholder
        self.train_loss = []
        self.val_loss = []
        self.epochs = []

    def on_epoch_end(self, epoch, logs=None):
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.epochs.append(epoch + 1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.epochs, y=self.train_loss, mode='lines+markers', name='Train Loss'))
        fig.add_trace(go.Scatter(x=self.epochs, y=self.val_loss, mode='lines+markers', name='Val Loss'))
        fig.update_layout(title="Training Progress", xaxis_title="Epoch", yaxis_title="Loss",
                          margin=dict(l=20, r=20, t=40, b=20))
        self.plot_placeholder.plotly_chart(fig, width="stretch")


# -----------------------------------------------------------------------------
# New Visualization Helpers
# -----------------------------------------------------------------------------

@st.cache_data
def get_training_stats(_model, X_train, y_train, label_map):
    """
    Pre-computes feature distributions and average spectra for the training set.
    """
    # X_train is scaled (mean=0, std=1), so RMS is ~1. 
    # We use Kurtosis (shape) and Max Amplitude (outlier strength) which survive scaling.
    X_flat = X_train.squeeze()  # (N, 1024)

    # 1. Feature Space
    # We need a subset for heavy SHAP calculations
    # Stratified sampling: 50 per class
    sample_indices = []
    unique_labels = np.unique(y_train)
    for lbl in unique_labels:
        indices = np.where(y_train == lbl)[0]
        if len(indices) > 50:
            sample_indices.extend(np.random.choice(indices, 50, replace=False))
        else:
            sample_indices.extend(indices)

    subset_X = X_train[sample_indices]
    subset_y = y_train[sample_indices]
    subset_flat = subset_X.squeeze()

    # Compute RMS (Root Mean Square)
    rms_vals = np.sqrt(np.mean(subset_flat ** 2, axis=1))

    # Compute SHAP Impact (Mean Absolute SHAP value)
    # Use a small background for speed
    n_bg = min(100, X_train.shape[0])
    background = X_train[np.random.choice(X_train.shape[0], n_bg, replace=False)]

    # explainer = shap.GradientExplainer(_model, background) # Can be slow/tricky with TF graph
    # Fallback to a simpler heuristic or just compute it if possible? 
    # Let's try to compute it. If it fails, we fall back to 0.
    shap_impacts = []

    if shap:
        try:
            # Note: GradientExplainer might need eager execution disabled or specific context
            # For robustness in thiscached function, we'll assume it works or returns 0
            explainer = shap.GradientExplainer(_model, background)
            shap_values = explainer.shap_values(subset_X)

            # shap_values is list [N, 1024, 1] per class
            # Aggregation: For each sample, we want the impact towards its *predicted* class (or true class?)
            # Let's just take the MAX impact across all classes for simplicity (strongest driver)

            # Combine absolute impacts across classes? 
            # Or just take the subset_X's predicted class impact?
            # Let's sum absolute values across the time dimension (Total Evidence)

            # If multi-class list
            if isinstance(shap_values, list):
                # Sum abs values for each class, then take max across classes?
                # This represents "How strongly does the model feel about ANY class?"
                class_impacts = [np.mean(np.abs(sv.squeeze()), axis=1) for sv in shap_values]  # (N,) per class
                shap_impacts = np.max(np.array(class_impacts), axis=0)  # (N,)
            else:
                shap_impacts = np.mean(np.abs(shap_values.squeeze()), axis=1)

        except Exception as e:
            st.error(f"SHAP calc failed in stats: {e}")
            shap_impacts = np.zeros(len(subset_X))
    else:
        shap_impacts = np.zeros(len(subset_X))

    # Compute Confidence for the subset
    # We do this here to ensure alignment with the subset
    subset_probs = _model.predict(subset_X, verbose=0)
    subset_conf = np.max(subset_probs, axis=1)

    # Ensure all arrays are 1D for DataFrame
    rms_vals = np.array(rms_vals).flatten()
    shap_impacts = np.array(shap_impacts).flatten()
    confidence_vals = np.array(subset_conf).flatten()
    labels_list = [label_map[y] for y in subset_y.flatten()]

    # Debug check (will print to console if running locally, or just prevent error)
    min_len = min(len(rms_vals), len(shap_impacts), len(confidence_vals), len(labels_list))

    features = pd.DataFrame({
        'RMS': rms_vals[:min_len],
        'SHAP_Impact': shap_impacts[:min_len],
        'Confidence': confidence_vals[:min_len],
        'Label': labels_list[:min_len]
    })

    # 2. Spectral Signatures (Average FFT per class)
    spectra = {}
    freqs = np.fft.rfftfreq(X_flat.shape[1])

    for lbl in unique_labels:
        # Get all signals for this class
        class_signals = X_flat[y_train == lbl]
        # Compute FFT magnitude
        fft_vals = np.abs(np.fft.rfft(class_signals, axis=1))
        # Average them
        avg_spec = np.mean(fft_vals, axis=0)
        spectra[label_map[lbl]] = avg_spec

    return features, spectra, freqs


def extract_current_features(windows):
    """Calculates features for the currently uploaded file."""
    # windows shape: (N, 1024)
    # Average the features over all windows in the file to get a "File Summary"

    # Per-window stats
    kurt_vals = kurtosis(windows, axis=1)
    max_vals = np.max(np.abs(windows), axis=1)
    rms_vals = np.sqrt(np.mean(windows ** 2, axis=1))

    # Centroid (Mean point for the whole file)
    avg_kurt = np.mean(kurt_vals)
    avg_max = np.mean(max_vals)
    avg_rms = np.mean(rms_vals)

    return avg_kurt, avg_max, avg_rms, kurt_vals, max_vals


def compute_current_spectrum(windows):
    """Calculates average spectrum for the currently uploaded file."""
    fft_vals = np.abs(np.fft.rfft(windows, axis=1))
    avg_spec = np.mean(fft_vals, axis=0)
    return avg_spec


# -----------------------------------------------------------------------------
# Mode A: Field Diagnostics (User)
# -----------------------------------------------------------------------------
def field_diagnostics_mode():
    st.title("🛡️ Field Diagnostics")
    st.markdown("### Daily Diagnostic Tool for Bearing Health")

    # --- Step 1: Onboarding ---
    with st.expander("📘 How to use this tool", expanded=False):
        st.markdown("""
        1.  **Download Template**: Get the standard CSV format.
        2.  **Record Data**: Paste your vibration sensor readings into the file.
        3.  **Upload & Analyze**: Upload the file below to get an instant health check.
        """)

    # --- Step 2: Data Template ---
    col_temp, _ = st.columns([1, 2])
    with col_temp:
        dummy_data = pd.DataFrame({"vibration_signal": np.random.normal(0, 1, 1500)})
        csv_template = dummy_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Data Template (.csv)",
            data=csv_template,
            file_name="sensor_data_template.csv",
            mime="text/csv"
        )

    st.divider()

    # --- Step 3: Upload & Inference ---
    st.subheader("Upload Sensor Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"],
                                     help="Upload a CSV file with at least 1024 sensor readings.")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            is_valid, msg, signal_values = validate_uploaded_file(df)

            if not is_valid:
                st.error(msg)
            else:
                st.success(msg)

                if 'model' not in st.session_state or 'scaler' not in st.session_state:
                    st.warning("⚠️ System Model Not Loaded. Please contact Admin to train/load the model first.")
                else:
                    with st.spinner("Running Diagnostics..."):
                        # Preprocessing
                        windows = create_sliding_windows(signal_values, WINDOW_SIZE, STRIDE)

                        if len(windows) == 0:
                            st.error(f"❌ Data too short. Need more than {WINDOW_SIZE} samples.")
                        else:
                            # Scale
                            windows = np.expand_dims(windows, axis=-1)
                            N, W, C = windows.shape
                            windows_reshaped = windows.reshape(-1, C)

                            # Note: We use the scaled windows for Prediction (Model expects scaled)
                            # But we also want "Raw-ish" characteristics. 
                            # Since X_train is stored as scaled, we will compare using scaled windows 
                            # to be mathematically consistent with the training data distribution.
                            windows_scaled = st.session_state.scaler.transform(windows_reshaped)
                            X_new = windows_scaled.reshape(N, W, C)

                            # Predict
                            pred_probs_all = st.session_state.model.predict(X_new)
                            pred_indices = np.argmax(pred_probs_all, axis=1)

                            # Aggregation (Majority Vote)
                            final_pred_idx = np.bincount(pred_indices).argmax()
                            final_label = st.session_state.label_map[final_pred_idx]

                            # Aggregate Confidence (Mean probability of the winning class)
                            avg_probs = np.mean(pred_probs_all, axis=0)
                            confidence = avg_probs[final_pred_idx]

                            # ---------------------------------------------------------
                            # RESULTS SECTION
                            # ---------------------------------------------------------
                            st.divider()
                            st.subheader("Diagnostic Result")

                            # 1. Main Alert
                            if final_label == "Normal":
                                st.success(f"🟢 **SYSTEM NORMAL** (Confidence: {confidence:.1%})")
                            else:
                                st.error(f"🔴 **CRITICAL FAULT: {final_label.upper()}** (Confidence: {confidence:.1%})")
                                st.warning("⚠️ Recommendation: Stop machine and inspect bearing immediately.")

                            # 2. Confidence Bar Chart
                            with st.expander("📊 Detailed Probability Analysis", expanded=True):
                                prob_df = pd.DataFrame({
                                    "Condition": list(st.session_state.label_map.values()),
                                    "Probability": avg_probs
                                })
                                fig_prob = px.bar(prob_df, x="Probability", y="Condition", orientation='h',
                                                  text_auto='.1%', title="Model Confidence per Class",
                                                  color="Condition", color_discrete_sequence=px.colors.qualitative.Bold)
                                st.plotly_chart(fig_prob, width="stretch")

                            # ---------------------------------------------------------
                            # EXPLAINABILITY DASHBOARD
                            # ---------------------------------------------------------
                            st.divider()
                            st.subheader("🔍 Why did the model make this decision?")



                            # --- Check if Training Data is available for comparisons ---
                            has_train_data = 'X_train' in st.session_state

                            if has_train_data:
                                # Pre-compute/Retrieve Reference Stats
                                # Pass model for SHAP calc
                                ref_features, _, _ = get_training_stats(
                                    st.session_state.model,
                                    st.session_state.X_train,
                                    st.session_state.y_train,
                                    st.session_state.label_map
                                )

                                # Compute Current Stats (on Scaled data to match X_train)
                                curr_windows_flat = X_new.squeeze()
                                curr_kurt, curr_max, curr_rms, _, _ = extract_current_features(curr_windows_flat)

                                # Compute SHAP Impact for Current File (on-demand)
                                # We need this for the X-coordinate in Tab 2
                                curr_shap_impact = 0
                                if has_train_data:
                                    # Pick best representative window (e.g. median RMS window to avoid outliers?)
                                    # Or just the one with highest prediction confidence
                                    shap_target_idx = np.argmax(np.max(pred_probs_all, axis=1))
                                    shap_target_window = X_new[shap_target_idx]

                                    sv, _ = compute_shap_explanation(
                                        st.session_state.model,
                                        st.session_state.X_train,
                                        shap_target_window
                                    )

                                    if sv is not None:
                                        if isinstance(sv, list):
                                            # Take max impact across classes
                                            class_impacts = [np.mean(np.abs(s.squeeze())) for s in sv]
                                            curr_shap_impact = np.max(class_impacts)
                                        else:
                                            curr_shap_impact = np.mean(np.abs(sv.squeeze()))

                                # --- FEATURE MAP (Comparison) ---
                                st.subheader("Feature Map (Comparison)")
                                st.caption("SHAP Impact vs. RMS Energy")

                                # 1. Tối ưu hóa Style
                                sns.set_theme(style="whitegrid")

                                # 2. Tạo Jointplot với mật độ
                                g = sns.jointplot(
                                    data=ref_features,
                                    x="SHAP_Impact",
                                    y="RMS",
                                    hue="Label",
                                    kind="scatter",
                                    palette="bright",
                                    height=8,
                                    alpha=0.4,  # Giảm alpha để nổi bật dấu X
                                    s=50  # Kích thước điểm reference vừa phải
                                )

                                # 3. Vẽ thêm đường đồng mức để xác định "vùng an toàn"
                                g.plot_joint(sns.kdeplot, alpha=0.2, zorder=1, levels=5, fill=False)

                                # 5. Overlay User's Point
                                g.ax_joint.scatter(
                                    [curr_shap_impact], [curr_rms],
                                    color='black', s=250, marker='X',
                                    label='CURRENT SIGNAL', zorder=10, linewidths=3,
                                    edgecolor='white'  # Thêm viền trắng để nổi bật hơn nữa
                                )

                                # 6. Cải thiện Annotation
                                g.ax_joint.text(
                                    curr_shap_impact, curr_rms + (0.02 * g.ax_joint.get_ylim()[1]),
                                    "📍 YOU ARE HERE",
                                    color='black', weight='bold', size=12,
                                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2)
                                    # Thêm nền cho text dễ đọc
                                )

                                # Labeling
                                g.ax_joint.set_xlabel("AI Sensitivity (SHAP Impact)")
                                g.ax_joint.set_ylabel("Vibration Energy (RMS)")

                                st.pyplot(g.fig)

                            else:
                                st.warning("⚠️ Training data required for Feature Map comparison.")



        except Exception as e:
            st.error(f"❌ Error parsing file: {e}")
            with st.expander("See Technical Error"):
                st.exception(e)


# -----------------------------------------------------------------------------
# Mode B: Model Training (Admin)
# -----------------------------------------------------------------------------
def model_training_mode():
    st.title("⚙️ Model Training & Administration")
    st.info("Restricted Area: For Data Scientists & Admins only.")

    tab1, tab2 = st.tabs(["1. Data & Training", "2. Metrics & Persistence"])

    with tab1:
        st.subheader("Data Pipeline")
        if st.button("🔄 Download & Process Data (Kaggle)"):
            data_path = download_dataset()
            X_train, X_test, y_train, y_test, label_map, scaler = load_and_process_data(data_path)

            if X_train is not None:
                st.session_state.update({
                    'X_train': X_train, 'X_test': X_test,
                    'y_train': y_train, 'y_test': y_test,
                    'label_map': label_map, 'scaler': scaler,
                    'data_loaded': True
                })
                st.success(f"✅ Data Loaded! Training Samples: {X_train.shape[0]}")
            else:
                st.error("Failed to process data.")

        if st.session_state.get('data_loaded'):
            st.divider()
            st.subheader("Train Model")
            col1, col2 = st.columns(2)
            epochs = col1.slider("Epochs", 1, 50, 10)
            batch_size = col2.select_slider("Batch", [16, 32, 64, 128], value=32)

            if st.button("🚀 Start Training"):
                input_shape = (st.session_state.X_train.shape[1], st.session_state.X_train.shape[2])
                num_classes = len(st.session_state.label_map)

                model = build_cnn_model(input_shape, num_classes)
                placeholder = st.empty()
                history = model.fit(
                    st.session_state.X_train, st.session_state.y_train,
                    validation_split=0.2, epochs=epochs, batch_size=batch_size,
                    callbacks=[StreamlitCallback(placeholder)], verbose=0
                )

                st.session_state.model = model
                st.session_state.history = history
                st.session_state.model_trained = True

                # --- NEW: Auto-Save to Disk ---
                model.save("bearing_model.keras")
                st.success("Training Complete! Model saved locally as 'bearing_model.keras'.")

    with tab2:
        if st.session_state.get('model_trained'):
            st.subheader("Evaluation Metrics")
            loss, acc = st.session_state.model.evaluate(st.session_state.X_test, st.session_state.y_test, verbose=0)
            col1, col2 = st.columns(2)
            col1.metric("Test Accuracy", f"{acc:.2%}")
            col2.metric("Test Loss", f"{loss:.4f}")

            y_pred = np.argmax(st.session_state.model.predict(st.session_state.X_test), axis=1)
            cm = confusion_matrix(st.session_state.y_test, y_pred)
            labels = list(st.session_state.label_map.values())
            fig = px.imshow(cm, text_auto=True, x=labels, y=labels, color_continuous_scale="Blues",
                            title="Confusion Matrix")
            st.plotly_chart(fig, width="stretch")

            st.divider()
            st.subheader("Model ersistence")
            with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
                model_path = tmp.name
            st.session_state.model.save(model_path)
            with open(model_path, "rb") as f:
                st.download_button("💾 Download .keras Model", f, "bearing_model.keras")


# -----------------------------------------------------------------------------
# Main & Navigation
# -----------------------------------------------------------------------------
def main():
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Go to", ["Field Diagnostics (User)", "Model Training (Admin)"])

    st.sidebar.divider()
    st.sidebar.info("Version: 2.2.0 (Persistence Enabled)")

    # --- NEW: Auto-Load Model on Startup ---
    # Only try to load if we haven't loaded it yet
    if 'model' not in st.session_state and os.path.exists("bearing_model.keras"):
        try:
            st.session_state.model = tf.keras.models.load_model("bearing_model.keras")
            # Note: We still need the Label Map and Scaler.
            # In a full app, you would save/load those using 'joblib'.
            # For this prototype, we re-initialize the label map.
            st.session_state.label_map = {0: "Normal", 1: "Inner Race Fault", 2: "Outer Race Fault"}

            # CRITICAL: We cannot easily restore the fitted StandardScaler without saving it.
            # Ideally, you should save the scaler using joblib.dump(scaler, 'scaler.bin')
            # For now, we will warn the user if the scaler is missing.
            if 'scaler' not in st.session_state:
                st.sidebar.warning(
                    "⚠️ Model loaded from disk, but Scaler is missing. Please go to Admin -> Download Data to restore the Scaler.")

            st.sidebar.success("✅ Saved model loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Failed to load saved model: {e}")

    # Routing
    if app_mode == "Field Diagnostics (User)":
        field_diagnostics_mode()
    else:
        model_training_mode()


if __name__ == "__main__":
    main()
