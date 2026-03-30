"""
preprocessing.py
----------------
Data acquisition, validation, and preprocessing pipeline
for the Intel Image Classification project.
"""

import os
import json
import shutil
import random
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Optional

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
CLASS_NAMES   = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
IMG_SIZE      = (150, 150)
BATCH_SIZE    = 32
VALID_EXTS    = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
SEED          = 42


# ── Validation ─────────────────────────────────────────────────────────────────
def validate_dataset_structure(data_dir: str) -> Dict:
    """
    Check that a directory contains expected class subfolders with images.
    Returns a report dict.
    """
    data_dir = Path(data_dir)
    report = {
        'valid': True,
        'path': str(data_dir),
        'classes_found': [],
        'missing_classes': [],
        'class_counts': {},
        'total_images': 0,
        'errors': []
    }

    if not data_dir.exists():
        report['valid'] = False
        report['errors'].append(f"Directory not found: {data_dir}")
        return report

    for cls in CLASS_NAMES:
        cls_dir = data_dir / cls
        if cls_dir.exists():
            imgs = [f for f in cls_dir.iterdir() if f.suffix.lower() in VALID_EXTS]
            report['classes_found'].append(cls)
            report['class_counts'][cls] = len(imgs)
            report['total_images'] += len(imgs)
            if len(imgs) == 0:
                report['errors'].append(f"Class '{cls}' has no valid images")
        else:
            report['missing_classes'].append(cls)
            report['errors'].append(f"Missing class directory: {cls}")

    if report['missing_classes']:
        report['valid'] = False

    return report


def validate_single_image(image_path: str) -> Tuple[bool, str]:
    """Validate a single image file. Returns (is_valid, message)."""
    path = Path(image_path)
    if not path.exists():
        return False, f"File not found: {image_path}"
    if path.suffix.lower() not in VALID_EXTS:
        return False, f"Unsupported format: {path.suffix}"
    try:
        img = tf.keras.preprocessing.image.load_img(str(path), target_size=IMG_SIZE)
        arr = tf.keras.preprocessing.image.img_to_array(img)
        if arr.shape != (*IMG_SIZE, 3):
            return False, f"Unexpected shape: {arr.shape}"
        return True, "Valid"
    except Exception as e:
        return False, f"Load error: {e}"


# ── Data generators ────────────────────────────────────────────────────────────
def get_train_datagen(augment: bool = True) -> ImageDataGenerator:
    """Return training ImageDataGenerator (with or without augmentation)."""
    if augment:
        return ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest',
            validation_split=0.15
        )
    return ImageDataGenerator(rescale=1./255, validation_split=0.15)


def get_test_datagen() -> ImageDataGenerator:
    """Return test/inference ImageDataGenerator (no augmentation)."""
    return ImageDataGenerator(rescale=1./255)


def build_generators(
    train_dir: str,
    test_dir: str,
    img_size: Tuple[int, int] = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    augment: bool = True
):
    """
    Build and return (train_gen, val_gen, test_gen).
    """
    train_datagen = get_train_datagen(augment)
    test_datagen  = get_test_datagen()

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        seed=SEED,
        shuffle=True,
        classes=CLASS_NAMES
    )
    val_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        seed=SEED,
        shuffle=False,
        classes=CLASS_NAMES
    )
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        classes=CLASS_NAMES
    )
    logger.info(f"Train: {train_gen.samples} | Val: {val_gen.samples} | Test: {test_gen.samples}")
    return train_gen, val_gen, test_gen


# ── New data ingestion ─────────────────────────────────────────────────────────
def ingest_uploaded_data(
    upload_dir: str,
    target_train_dir: str,
    validate_only: bool = False
) -> Dict:
    """
    Move uploaded images from upload_dir into the training directory.
    Expects upload_dir to have class subfolders matching CLASS_NAMES.

    Returns:
        Dict with ingestion report
    """
    upload_dir  = Path(upload_dir)
    target_dir  = Path(target_train_dir)
    report = {
        'timestamp': datetime.now().isoformat(),
        'validated': False,
        'ingested': False,
        'class_counts': {},
        'total_ingested': 0,
        'errors': []
    }

    # Validate
    val_report = validate_dataset_structure(str(upload_dir))
    report['validation'] = val_report

    if not val_report['valid']:
        report['errors'] = val_report['errors']
        return report

    report['validated'] = True

    if validate_only:
        return report

    # Copy images
    for cls in CLASS_NAMES:
        src_dir = upload_dir / cls
        dst_dir = target_dir / cls
        dst_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        for img_path in src_dir.iterdir():
            if img_path.suffix.lower() in VALID_EXTS:
                is_valid, msg = validate_single_image(str(img_path))
                if is_valid:
                    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
                    dst_path = dst_dir / f"{timestamp}_{img_path.name}"
                    shutil.copy2(str(img_path), str(dst_path))
                    copied += 1
                else:
                    report['errors'].append(f"{img_path.name}: {msg}")

        report['class_counts'][cls] = copied
        report['total_ingested'] += copied

    report['ingested'] = True
    logger.info(f"Ingested {report['total_ingested']} images into {target_dir}")
    return report


# ── Image preprocessing ────────────────────────────────────────────────────────
def preprocess_image(image_path: str) -> Optional[np.ndarray]:
    """Load and preprocess a single image for inference."""
    try:
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
        arr = tf.keras.preprocessing.image.img_to_array(img)
        arr = arr / 255.0
        return np.expand_dims(arr, axis=0)
    except Exception as e:
        logger.error(f"Failed to preprocess {image_path}: {e}")
        return None


def preprocess_image_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    """Load and preprocess image from raw bytes for API inference."""
    import io
    from PIL import Image
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize(IMG_SIZE)
        arr = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0)
    except Exception as e:
        logger.error(f"Failed to preprocess image bytes: {e}")
        return None


if __name__ == '__main__':
    # Quick self-test
    print("Preprocessing module loaded successfully.")
    print(f"Class names: {CLASS_NAMES}")
    print(f"Image size:  {IMG_SIZE}")
    print(f"Batch size:  {BATCH_SIZE}")
