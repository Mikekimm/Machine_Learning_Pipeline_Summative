"""
model.py
--------
Model architecture, training, fine-tuning, and retraining logic
for the Intel Image Classification project.
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)

from preprocessing import (
    CLASS_NAMES, IMG_SIZE, BATCH_SIZE, SEED, build_generators
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

MODELS_DIR   = Path('../models')
NUM_CLASSES  = len(CLASS_NAMES)
MODELS_DIR.mkdir(exist_ok=True)


# ── Architecture ───────────────────────────────────────────────────────────────
def build_model(
    num_classes: int  = NUM_CLASSES,
    img_size: Tuple   = IMG_SIZE,
    trainable_base: bool = False,
    dropout_rate: float  = 0.4
) -> Tuple[Model, Model]:
    """
    Build MobileNetV2-based image classifier.

    Returns:
        (full_model, base_model) — base_model exposed so layers can be unfrozen.
    """
    tf.random.set_seed(SEED)

    base_model = MobileNetV2(
        input_shape=(*img_size, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = trainable_base

    inputs = keras.Input(shape=(*img_size, 3), name='image_input')
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.BatchNormalization(name='bn')(x)
    x = layers.Dense(256, activation='relu', name='dense1')(x)
    x = layers.Dropout(dropout_rate, name='drop1')(x)
    x = layers.Dense(128, activation='relu', name='dense2')(x)
    x = layers.Dropout(dropout_rate * 0.75, name='drop2')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = Model(inputs, outputs, name='IntelClassifier_MobileNetV2')
    logger.info(f"Model built | Total params: {model.count_params():,}")
    return model, base_model


def compile_model(
    model: Model,
    learning_rate: float = 1e-3
) -> Model:
    """Compile model with Adam + categorical crossentropy."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    return model


def get_callbacks(checkpoint_path: str) -> list:
    """Standard training callbacks."""
    return [
        EarlyStopping(
            monitor='val_accuracy', patience=5,
            restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            checkpoint_path, monitor='val_accuracy',
            save_best_only=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3,
            min_lr=1e-8, verbose=1
        )
    ]


# ── Training ───────────────────────────────────────────────────────────────────
def train(
    train_dir: str,
    test_dir: str,
    epochs_phase1: int = 15,
    epochs_phase2: int = 10,
    batch_size: int = BATCH_SIZE,
    save_dir: str = str(MODELS_DIR)
) -> Dict:
    """
    Full two-phase training:
      Phase 1 — frozen base (feature extraction)
      Phase 2 — top-30 layers unfrozen (fine-tuning)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    train_gen, val_gen, test_gen = build_generators(train_dir, test_dir, batch_size=batch_size)

    # ── Phase 1 ──
    logger.info("Phase 1: Training with frozen base...")
    model, base_model = build_model(trainable_base=False)
    model = compile_model(model, learning_rate=1e-3)
    cbs = get_callbacks(str(save_dir / 'intel_classifier_best.h5'))

    history1 = model.fit(
        train_gen, epochs=epochs_phase1,
        validation_data=val_gen, callbacks=cbs, verbose=1
    )

    # ── Phase 2 ──
    logger.info("Phase 2: Fine-tuning top layers...")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model = compile_model(model, learning_rate=1e-5)
    history2 = model.fit(
        train_gen, epochs=epochs_phase2,
        validation_data=val_gen, callbacks=cbs, verbose=1
    )

    # ── Save final ──
    final_path = str(save_dir / 'intel_classifier_final.h5')
    model.save(final_path)
    logger.info(f"Final model saved → {final_path}")

    # ── Evaluate ──
    test_gen.reset()
    results = model.evaluate(test_gen, verbose=0)
    metrics = dict(zip(model.metrics_names, results))

    # ── Metadata ──
    metadata = {
        'model_name': 'IntelImageClassifier_MobileNetV2',
        'version': '1.0.0',
        'created_at': datetime.now().isoformat(),
        'architecture': 'MobileNetV2 + Custom Head',
        'class_names': CLASS_NAMES,
        'input_size': list(IMG_SIZE) + [3],
        'metrics': {k: float(v) for k, v in metrics.items()}
    }
    with open(str(save_dir / 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Test metrics: {metrics}")
    return {'history1': history1.history, 'history2': history2.history, 'metrics': metrics}


# ── Retraining ─────────────────────────────────────────────────────────────────
def retrain(
    new_data_dir: str,
    model_path: str      = str(MODELS_DIR / 'intel_classifier_final.h5'),
    epochs: int          = 5,
    learning_rate: float = 1e-5,
    save_dir: str        = str(MODELS_DIR)
) -> Dict:
    """
    Retrain an existing model on new data.
    Trigger this when new data has been uploaded and validated.

    Args:
        new_data_dir:   Directory with new images in class subfolders
        model_path:     Path to the base model to retrain
        epochs:         Max retraining epochs (EarlyStopping may cut short)
        learning_rate:  Low LR to avoid catastrophic forgetting
        save_dir:       Where to save the retrained model

    Returns:
        Dict with history, new model path, and metrics
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    logger.info(f"[RETRAIN] Loading base model: {model_path}")
    model = keras.models.load_model(model_path)

    # Use a minimal augmentation for retraining
    from preprocessing import get_train_datagen, get_test_datagen
    datagen = get_train_datagen(augment=True)

    train_gen = datagen.flow_from_directory(
        new_data_dir, target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode='categorical',
        subset='training', seed=SEED, classes=CLASS_NAMES
    )
    val_gen = datagen.flow_from_directory(
        new_data_dir, target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode='categorical',
        subset='validation', seed=SEED, classes=CLASS_NAMES
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )

    timestamp    = datetime.now().strftime('%Y%m%d_%H%M%S')
    retrain_path = str(save_dir / f'intel_classifier_retrained_{timestamp}.h5')

    cbs = [
        EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, verbose=1),
        ModelCheckpoint(retrain_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    logger.info(f"[RETRAIN] Training for up to {epochs} epochs...")
    history = model.fit(
        train_gen, epochs=epochs,
        validation_data=val_gen, callbacks=cbs, verbose=1
    )

    # Update metadata
    meta_path = save_dir / 'model_metadata.json'
    if meta_path.exists():
        with open(str(meta_path)) as f:
            metadata = json.load(f)
    else:
        metadata = {}

    metadata['last_retrained_at'] = datetime.now().isoformat()
    metadata['retrain_data_dir']  = new_data_dir
    metadata['retrain_epochs']    = len(history.history['loss'])
    metadata['retrain_val_acc']   = float(max(history.history['val_accuracy']))

    with open(str(meta_path), 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"[RETRAIN] Complete. Model saved → {retrain_path}")
    logger.info(f"[RETRAIN] Best val_accuracy: {metadata['retrain_val_acc']:.4f}")

    return {
        'history': history.history,
        'retrained_model_path': retrain_path,
        'best_val_accuracy': metadata['retrain_val_acc']
    }


# ── Retraining trigger ─────────────────────────────────────────────────────────
RETRAIN_TRIGGER_FILE = str(MODELS_DIR / 'retrain_trigger.json')

def check_retrain_trigger() -> Optional[Dict]:
    """
    Check if a retraining trigger file exists.
    Returns trigger metadata if found, else None.
    Used by background worker / scheduler.
    """
    p = Path(RETRAIN_TRIGGER_FILE)
    if p.exists():
        with open(str(p)) as f:
            return json.load(f)
    return None

def set_retrain_trigger(data_dir: str, reason: str = 'manual') -> None:
    """Write a retraining trigger file (called from API/UI)."""
    trigger = {
        'triggered_at': datetime.now().isoformat(),
        'data_dir': data_dir,
        'reason': reason,
        'status': 'pending'
    }
    with open(RETRAIN_TRIGGER_FILE, 'w') as f:
        json.dump(trigger, f, indent=2)
    logger.info(f"[TRIGGER] Retraining trigger set: {trigger}")

def clear_retrain_trigger() -> None:
    """Remove trigger file after retraining completes."""
    p = Path(RETRAIN_TRIGGER_FILE)
    if p.exists():
        p.unlink()


if __name__ == '__main__':
    model, base = build_model()
    model.summary()
    print(f"\nTrainable parameters: {sum([np.prod(v.shape) for v in model.trainable_variables]):,}")
