"""
api/main.py
-----------
FastAPI REST API for Intel Image Classifier.

Endpoints:
  GET  /                   Health check
  GET  /health             Detailed health + model info
  GET  /model/info         Model metadata
  POST /predict            Single image prediction
  POST /predict/batch      Batch image prediction
  POST /upload             Upload new training data
  POST /retrain/trigger    Trigger model retraining
  GET  /retrain/status     Check retraining status
  GET  /metrics            Prometheus-style metrics
"""

import io
import os
import json
import time
import uuid
import shutil
import logging
import asyncio
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, List

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from prediction import (
    predict_from_bytes, predict_batch, warmup,
    get_model_info, get_active_model_path, reload_model
)
from preprocessing import validate_single_image, ingest_uploaded_data
from model import retrain, set_retrain_trigger, check_retrain_trigger, clear_retrain_trigger

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ── Paths ───────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.parent
MODELS_DIR   = BASE_DIR / 'models'
DATA_DIR     = BASE_DIR / 'data'
UPLOAD_DIR   = BASE_DIR / 'data' / 'uploads'
TRAIN_DIR    = BASE_DIR / 'data' / 'train'
TEST_DIR     = BASE_DIR / 'data' / 'test'
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── App ─────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Intel Image Classifier API",
    description="End-to-end ML pipeline for 6-class scene classification",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ────────────────────────────────────────────────────────────────
request_counter  = 0
error_counter    = 0
total_latency_ms = 0.0
uptime_start     = time.time()
retrain_status   = {'status': 'idle', 'last_run': None, 'result': None}


# ── Startup ─────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("API starting — warming up model...")
    try:
        result = warmup()
        logger.info(f"Warmup: {result}")
    except Exception as e:
        logger.warning(f"Warmup failed (model may not exist yet): {e}")


# ── Health & info ───────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "service": "Intel Image Classifier",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    uptime_s = time.time() - uptime_start
    try:
        model_info = get_model_info()
        model_status = "loaded"
    except Exception as e:
        model_info = {}
        model_status = f"error: {e}"

    return {
        "status": "healthy",
        "uptime_seconds": round(uptime_s, 1),
        "model_status": model_status,
        "model_path": get_active_model_path(),
        "requests_served": request_counter,
        "error_count": error_counter,
        "avg_latency_ms": round(total_latency_ms / max(request_counter, 1), 2),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model/info")
async def model_info():
    try:
        return get_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Prediction ──────────────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict class for a single uploaded image."""
    global request_counter, error_counter, total_latency_ms

    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        start = time.perf_counter()
        result = predict_from_bytes(image_bytes)
        elapsed = (time.perf_counter() - start) * 1000

        request_counter  += 1
        total_latency_ms += elapsed

        result['filename'] = file.filename
        return result

    except Exception as e:
        error_counter += 1
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch_endpoint(files: List[UploadFile] = File(...)):
    """Predict classes for multiple uploaded images."""
    global request_counter, error_counter

    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Max 50 images per batch")

    tmp_dir = UPLOAD_DIR / f"batch_{uuid.uuid4().hex[:8]}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        saved_paths = []
        for f in files:
            if not f.content_type.startswith('image/'):
                continue
            dst = tmp_dir / f.filename
            content = await f.read()
            dst.write_bytes(content)
            saved_paths.append(str(dst))

        results = predict_batch(saved_paths)
        request_counter += len(results)
        return {'count': len(results), 'predictions': results}

    except Exception as e:
        error_counter += 1
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)


# ── Data upload ─────────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload_data(
    files: List[UploadFile] = File(...),
    class_name: Optional[str] = None
):
    """
    Upload new training images.
    Files should be named as <class>/<filename>.jpg OR
    pass class_name query param for all files in this request.
    """
    valid_classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

    session_dir = UPLOAD_DIR / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    errors = []

    for f in files:
        if not f.content_type.startswith('image/'):
            errors.append(f"{f.filename}: not an image")
            continue

        # Determine class from filename path or query param
        fname = f.filename
        cls = class_name
        if not cls:
            parts = fname.replace('\\', '/').split('/')
            if len(parts) >= 2 and parts[-2] in valid_classes:
                cls = parts[-2]
                fname = parts[-1]

        if cls not in valid_classes:
            errors.append(f"{f.filename}: unknown class '{cls}'")
            continue

        cls_dir = session_dir / cls
        cls_dir.mkdir(exist_ok=True)
        dst = cls_dir / f"{uuid.uuid4().hex[:8]}_{fname}"
        content = await f.read()
        dst.write_bytes(content)
        saved.append({'file': f.filename, 'class': cls, 'saved_as': dst.name})

    return {
        'session_dir': str(session_dir),
        'saved_count': len(saved),
        'error_count': len(errors),
        'saved': saved,
        'errors': errors
    }


# ── Retraining ──────────────────────────────────────────────────────────────────
class RetrainRequest(BaseModel):
    data_dir:      Optional[str] = None   # if None, uses latest upload session
    epochs:        int = 5
    learning_rate: float = 1e-5
    reason:        str = 'manual_ui_trigger'


def _run_retrain(data_dir: str, epochs: int, lr: float):
    """Background retraining task."""
    global retrain_status
    retrain_status['status'] = 'running'
    retrain_status['started_at'] = datetime.now().isoformat()

    try:
        result = retrain(
            new_data_dir=data_dir,
            epochs=epochs,
            learning_rate=lr,
            save_dir=str(MODELS_DIR)
        )
        reload_model()   # Flush model cache
        clear_retrain_trigger()
        retrain_status['status'] = 'completed'
        retrain_status['last_run'] = datetime.now().isoformat()
        retrain_status['result'] = {
            'best_val_accuracy': result['best_val_accuracy'],
            'model_path': result['retrained_model_path']
        }
        logger.info(f"Retraining complete: {result['best_val_accuracy']:.4f}")
    except Exception as e:
        retrain_status['status'] = 'failed'
        retrain_status['error'] = str(e)
        logger.error(f"Retraining failed: {e}")


@app.post("/retrain/trigger")
async def trigger_retrain(
    background_tasks: BackgroundTasks,
    req: RetrainRequest = RetrainRequest()
):
    """Trigger model retraining in the background."""
    global retrain_status

    if retrain_status['status'] == 'running':
        raise HTTPException(status_code=409, detail="Retraining already in progress")

    # Find data dir
    data_dir = req.data_dir
    if not data_dir:
        sessions = sorted(UPLOAD_DIR.glob('session_*'))
        if not sessions:
            raise HTTPException(status_code=400, detail="No uploaded data found. Upload data first.")
        data_dir = str(sessions[-1])

    if not Path(data_dir).exists():
        raise HTTPException(status_code=404, detail=f"Data directory not found: {data_dir}")

    set_retrain_trigger(data_dir, req.reason)
    background_tasks.add_task(_run_retrain, data_dir, req.epochs, req.learning_rate)

    retrain_status['status'] = 'queued'
    retrain_status['data_dir'] = data_dir

    return {
        'message': 'Retraining triggered',
        'data_dir': data_dir,
        'epochs': req.epochs,
        'learning_rate': req.learning_rate,
        'check_status_at': '/retrain/status'
    }


@app.get("/retrain/status")
async def retrain_status_endpoint():
    """Check current retraining status."""
    return retrain_status


# ── Metrics ─────────────────────────────────────────────────────────────────────
@app.get("/metrics")
async def metrics():
    """Prometheus-style metrics for monitoring."""
    uptime_s = time.time() - uptime_start
    return {
        'uptime_seconds': round(uptime_s, 1),
        'requests_total': request_counter,
        'errors_total': error_counter,
        'avg_latency_ms': round(total_latency_ms / max(request_counter, 1), 2),
        'model_path': get_active_model_path(),
        'retrain_status': retrain_status['status'],
        'timestamp': datetime.now().isoformat()
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000, reload=False)
