"""
SignBridge — FastAPI Backend
Main application with CORS, health check, and lifespan model loading.
"""
import os
import sys
import time
from contextlib import asynccontextmanager

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.config import MODEL_DIR, VOCABULARY_DIR, ALLOWED_ORIGINS
from backend.services.classifier import load_models, get_models_status, get_model_labels
from backend.services.vocabulary import load_registry, get_registry, validate_models_and_registry
from backend.routers import recognize, speak, captions

_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and vocabulary on startup."""
    print("=" * 60)
    print("SignBridge Backend Starting...")
    print("=" * 60)
    
    # Load models
    model_dir = os.path.normpath(MODEL_DIR)
    print(f"Loading models from: {model_dir}")
    load_models(model_dir)
    
    # Load vocabulary registry
    registry_path = os.path.join(VOCABULARY_DIR, "registry.json")
    print(f"Loading vocabulary from: {registry_path}")
    load_registry(registry_path)
    
    # Validate registry classes match models
    labels = get_model_labels()
    validate_models_and_registry(labels)
    
    print("=" * 60)
    print("SignBridge Backend Ready!")
    print("=" * 60)
    
    yield
    
    print("SignBridge Backend Shutting Down...")


app = FastAPI(
    title="SignBridge API",
    description="Assistive Sign-Language-to-Speech Translator for Public Service Counters",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS + ["*"],  # Permissive for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for sign assets
signs_dir = os.path.join(os.path.dirname(__file__), "..", "frontend", "public", "signs")
if not os.path.isdir(signs_dir):
    signs_dir = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist", "signs")
if os.path.isdir(signs_dir):
    app.mount("/signs", StaticFiles(directory=signs_dir), name="signs")

# Include routers
app.include_router(recognize.router, prefix="/api")
app.include_router(speak.router, prefix="/api")
app.include_router(captions.router, prefix="/api")


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    models = get_models_status()
    return {
        "status": "healthy",
        "models_loaded": models,
        "version": "1.0.0",
        "uptime_s": round(time.time() - _start_time, 1)
    }


@app.get("/api/vocabulary")
async def vocabulary():
    """Return the full vocabulary registry."""
    return get_registry()


# Mount production frontend dist if available, else API fallback
frontend_dist = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))
if os.path.isdir(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
else:
    @app.get("/")
    async def root():
        return {
            "app": "SignBridge",
            "description": "Assistive Sign-Language-to-Speech Translator",
            "docs": "/docs",
            "health": "/api/health"
        }

