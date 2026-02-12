import os

# Configuration Constants
WINDOW_SIZE = 1024
STRIDE = 512
TEST_SIZE = 0.2
RANDOM_SEED = 42

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Data and models should be in the 'ai_service' root (one level up from 'core')
AI_SERVICE_ROOT = os.path.dirname(BASE_DIR)

DATASET_PATH = os.path.join(AI_SERVICE_ROOT, "data/dataset")
MODEL_PATH = os.path.join(AI_SERVICE_ROOT, "bearing_model.keras")
SCALER_PATH = os.path.join(AI_SERVICE_ROOT, "scaler.joblib")
SHAP_BACKGROUND_PATH = os.path.join(AI_SERVICE_ROOT, "background_data.joblib")
FEATURE_MODEL_PATH = os.path.join(AI_SERVICE_ROOT, "feature_model.joblib")
HISTORY_PATH = os.path.join(AI_SERVICE_ROOT, "last_training_history.joblib")
