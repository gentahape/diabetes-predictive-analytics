import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import (
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
    CORS_ORIGINS,
    ARTIFACTS_DIR,
)
from .schemas import (
    PatientInput,
    PredictionResponse,
    HealthResponse,
    MetadataResponse,
    PresetsResponse,
)
from .services import ModelService, get_model_service

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("diabetes_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        service = get_model_service()
    except Exception as exc:
        logger.critical(f"Failed to load ML artifacts during startup: {exc}", exc_info=True)
        raise exc
    yield
    logger.info("Shutting down Diabetes Predictive Analytics API service...")

# Initialize FastAPI application
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
async def root():
    return {
        "message": "Diabetes Predictive Analytics REST API",
        "version": API_VERSION,
        "documentation": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    try:
        service = get_model_service()
        model_ready = service.model is not None
        scaler_ready = service.scaler is not None

        return HealthResponse(
            status="healthy" if (model_ready and scaler_ready) else "degraded",
            version=API_VERSION,
            model_loaded=model_ready,
            scaler_loaded=scaler_ready,
            artifacts_directory=str(ARTIFACTS_DIR),
        )
    except Exception as exc:
        logger.error(f"Health check failed: {exc}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "version": API_VERSION,
                "model_loaded": False,
                "scaler_loaded": False,
                "error": str(exc),
            },
        )


@app.get("/api/metadata", response_model=MetadataResponse, tags=["ML Metadata"])
async def get_metadata():
    service = get_model_service()
    if not service.metadata:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model metadata is not available.",
        )
    return service.metadata


@app.get("/api/presets", response_model=PresetsResponse, tags=["Presets"])
async def get_presets():
    service = get_model_service()
    if not service.presets:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Presets configuration is not available.",
        )
    return PresetsResponse(presets=service.presets)


@app.post(
    "/api/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
)
async def predict_diabetes_risk(patient: PatientInput):
    try:
        service = get_model_service()
        response = service.predict(patient)
        return response
    except Exception as exc:
        logger.error(f"Inference error encountered: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing prediction pipeline: {str(exc)}",
        )
