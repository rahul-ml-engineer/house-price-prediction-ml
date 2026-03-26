from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

TRAIN_DATA_PATH = RAW_DATA_DIR / "train.csv"
TEST_DATA_PATH = RAW_DATA_DIR / "test.csv"

PROCESSED_TRAIN_PATH = PROCESSED_DATA_DIR / "train_processed.csv"
FEATURES_TRAIN_PATH = PROCESSED_DATA_DIR / "train_features.csv"

# Models path
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "house_price_model.pkl"