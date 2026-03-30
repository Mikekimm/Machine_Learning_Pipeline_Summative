"""
prediction.py
-------------
Inference utilities for the Intel Image Classifier.
Supports single image, batch, and bytes-based prediction.
"""

import io
import json
import logging
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

import tensorflow as tf
from tensorflow import keras

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

CLASS_NAMES  = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
IMG_SIZE     = (150, 150)
MODELS_DIR   = Path('../models')
DEFAULT_MODEL = str(MODELS_DIR / 'intel_classifier_final.h5')


# ── Model loader (cached) ──────────────────────────────────────────────────────
_model_cache: Dict[str, keras.Model] = {}

def load_model(model_path: str = DEFAULT_MODEL) -> keras.Model:
    """Load model from disk, with in-process caching."""
    if model_path not in _model_cache:
        logger.info(f"Loading model: {model_path}")
        _model_cache[model_path] = keras.models.load_model(model_path)
        logger.info(f"Model loaded. Params: {_model_cache[model_path].count_params():,}")
    return _model_cache[model_path]

def reload_model(model_path: str = DEFAULT_MODEL) -> keras.Model:
    """Force-reload model (use after retraining)."""
    if model_path in _model_cache:
        del _model_cache[model_path]
    return load_model(model_path)

def get_active_model_path() -> str:
    """Return path to the most recent model."""
    retrained = sorted(MODELS_DIR.glob('intel_classifier_retrained_*.h5'))
    if retrained:
        return str(retrained[-1])
    final = MODELS_DIR / 'intel_classifier_final.h5'
    if final.exists():
        return str(final)
    return DEFAULT_MODEL


# ── Core prediction ────────────────────────────────────────────────────────────
def _preprocess_array(img_array: np.ndarray) -> np.ndarray:
    """Resize and normalise a numpy image array."""
    img = tf.image.resize(img_array, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img.numpy()

def predict_from_array(
    img_array: np.ndarray,
    model_path: Optional[str] = None
) -> Dict:
    """Predict from a numpy HWC image array (uint8 or float)."""
    model_path = model_path or get_active_model_path()
    model = load_model(model_path)

    processed = _preprocess_array(img_array)
    batch = np.expand_dims(processed, axis=0)

    start = time.perf_counter()
    probs = model.predict(batch, verbose=0)[0]
    latency_ms = (time.perf_counter() - start) * 1000

    pred_idx = int(np.argmax(probs))
    return {
        'predicted_class': CLASS_NAMES[pred_idx],
        'confidence': float(probs[pred_idx]),
        'all_probabilities': {cls: float(p) for cls, p in zip(CLASS_NAMES, probs)},
        'latency_ms': round(latency_ms, 2),
        'model_path': model_path,
        'timestamp': datetime.now().isoformat()
    }


def predict_from_path(
    image_path: str,
    model_path: Optional[str] = None
) -> Dict:
    """Predict from an image file path."""
    img = tf.keras.preprocessing.image.load_img(image_path)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    result = predict_from_array(img_array, model_path)
    result['image_path'] = image_path
    return result


def predict_from_bytes(
    image_bytes: bytes,
    model_path: Optional[str] = None
) -> Dict:
    """Predict from raw image bytes (e.g., from HTTP upload)."""
    from PIL import Image
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_array = np.array(img, dtype=np.uint8)
    return predict_from_array(img_array, model_path)


def predict_batch(
    image_paths: List[str],
    model_path: Optional[str] = None,
    batch_size: int = 32
) -> List[Dict]:
    """
    Predict a list of image file paths efficiently in batches.
    Returns a list of prediction dicts in the same order.
    """
    model_path = model_path or get_active_model_path()
    model = load_model(model_path)

    results = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_arrays = []

        for p in batch_paths:
            try:
                img = tf.keras.preprocessing.image.load_img(p)
                arr = tf.keras.preprocessing.image.img_to_array(img)
                batch_arrays.append(_preprocess_array(arr))
            except Exception as e:
                logger.warning(f"Skipping {p}: {e}")
                batch_arrays.append(np.zeros((*IMG_SIZE, 3), dtype=np.float32))

        batch_tensor = np.array(batch_arrays)
        start  = time.perf_counter()
        probs_batch = model.predict(batch_tensor, verbose=0)
        latency_ms  = (time.perf_counter() - start) * 1000 / len(batch_paths)

        for path, probs in zip(batch_paths, probs_batch):
            pred_idx = int(np.argmax(probs))
            results.append({
                'image_path': path,
                'predicted_class': CLASS_NAMES[pred_idx],
                'confidence': float(probs[pred_idx]),
                'all_probabilities': {cls: float(p) for cls, p in zip(CLASS_NAMES, probs)},
                'latency_ms': round(latency_ms, 2)
            })

    return results


# ── Health / warmup ────────────────────────────────────────────────────────────
def warmup(model_path: Optional[str] = None) -> Dict:
    """
    Run a dummy prediction to initialise TF graph.
    Call once at API startup to reduce first-request latency.
    """
    model_path = model_path or get_active_model_path()
    dummy = np.zeros((1, *IMG_SIZE, 3), dtype=np.float32)
    model = load_model(model_path)
    start = time.perf_counter()
    _ = model.predict(dummy, verbose=0)
    elapsed = (time.perf_counter() - start) * 1000
    logger.info(f"Warmup complete in {elapsed:.1f} ms")
    return {'status': 'warm', 'warmup_ms': round(elapsed, 2), 'model_path': model_path}


def get_model_info(model_path: Optional[str] = None) -> Dict:
    """Return model metadata."""
    model_path = model_path or get_active_model_path()
    meta_path  = MODELS_DIR / 'model_metadata.json'
    metadata   = {}
    if meta_path.exists():
        with open(str(meta_path)) as f:
            metadata = json.load(f)

    model = load_model(model_path)
    return {
        'model_path': model_path,
        'total_params': model.count_params(),
        'input_shape': list(model.input_shape[1:]),
        'output_classes': CLASS_NAMES,
        'metadata': metadata
    }


if __name__ == '__main__':
    info = get_model_info()
    print(json.dumps(info, indent=2, default=str))
