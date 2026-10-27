import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Base directory for the backend app
APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

# Load environment variables from .env
load_dotenv(BACKEND_DIR / ".env")

# Artifacts directory resolution
_env_artifacts = os.getenv("ARTIFACTS_DIR")
if _env_artifacts and Path(_env_artifacts).exists():
    ARTIFACTS_DIR = Path(_env_artifacts).resolve()
elif (PROJECT_ROOT / "notebook" / "artifacts").exists():
    ARTIFACTS_DIR = PROJECT_ROOT / "notebook" / "artifacts"
elif (BACKEND_DIR / "artifacts").exists():
    ARTIFACTS_DIR = BACKEND_DIR / "artifacts"
elif (PROJECT_ROOT / "artifacts").exists():
    ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
else:
    ARTIFACTS_DIR = PROJECT_ROOT / "notebook" / "artifacts"

# Artifact file paths
MODEL_PATH = ARTIFACTS_DIR / "lgbm_model.joblib"
SCALER_PATH = ARTIFACTS_DIR / "scaler.joblib"
FEATURE_COLUMNS_PATH = ARTIFACTS_DIR / "feature_columns.json"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
PRESETS_PATH = ARTIFACTS_DIR / "presets.json"

# API metadata
API_TITLE = "Diabetes Predictive Analytics API"
API_DESCRIPTION = (
    "Production-ready REST API for early detection and multi-class risk prediction "
    "of diabetes stages (No Diabetes, Pre-Diabetes, Diabetes) powered by an optimized LightGBM model."
)
API_VERSION = "1.0.0"

# CORS configuration
CORS_ORIGINS: List[str] = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "*",
]
