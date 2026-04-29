"""
EfficientNet-B7 embedding network via TensorFlow/Keras.

Weights: Happywhale Kaggle competition model (yellowdolphin/happywhale-models).
Source:  https://huggingface.co/yellowdolphin/happywhale-models
Code:    https://github.com/yellowdolphin/deeptrane (models_tf.py)

Architecture (efnv1b7_colab216, inference mode):
  - EfficientNet-B7 backbone  (Noisy Student pretrained, fine-tuned on cetaceans)
  - GlobalAveragePooling2D    →  2560-D feature vector
  - Dropout(0.2)              (identity at inference)
  - L2 normalisation          →  unit embedding

The h5 file is weights-only (saved with model.save_weights(), not model.save()).
We rebuild the inference architecture using the efficientnet package — whose layer
names match those in the saved file — then load with by_name=True.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np
import tensorflow as tf

MODEL_PATH = Path(os.getenv("MODEL_PATH", "./models/efnv1b7_colab216.h5"))
_MODEL_URL = (
    "https://huggingface.co/yellowdolphin/happywhale-models"
    "/resolve/main/efnv1b7_colab216.h5"
)
_IMAGE_SIZE = 600

# Set after first load.
EMBEDDING_DIM: int = 0

_embedding_model: tf.keras.Model | None = None


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _download_model_if_needed() -> None:
    """Stream the competition weights from HuggingFace if not on disk."""
    if MODEL_PATH.exists():
        return

    print(f"[model] Weights not found at {MODEL_PATH}")
    print(f"[model] Downloading from HuggingFace (~459 MB) …")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    def _report(block_num: int, block_size: int, total_size: int) -> None:
        if total_size > 0 and block_num % 500 == 0:
            pct = min(100, block_num * block_size * 100 // total_size)
            mb  = block_num * block_size // 1_048_576
            print(f"[model]   {pct}% ({mb} MB)", flush=True)

    urllib.request.urlretrieve(_MODEL_URL, MODEL_PATH, reporthook=_report)
    print(f"[model] Saved to {MODEL_PATH}")


# ---------------------------------------------------------------------------
# Build + load
# ---------------------------------------------------------------------------

def _build_and_load(weights_path: Path) -> tf.keras.Model:
    """
    Reconstruct the inference-mode architecture using the efficientnet package
    (whose layer names match those in the h5 file), then load weights by name.
    """
    import efficientnet.tfkeras as efn  # noqa: F401  (registers custom objects)

    # Build backbone with no pre-trained weights — we'll load from h5 instead.
    backbone = efn.EfficientNetB7(
        input_shape=(_IMAGE_SIZE, _IMAGE_SIZE, 3),
        weights=None,
        include_top=False,
    )

    inp = backbone.input
    x   = backbone.output
    x   = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x   = tf.keras.layers.Dropout(0.2, name="dropout")(x)
    out = tf.keras.layers.Lambda(
        lambda v: tf.math.l2_normalize(v, axis=1),
        name="l2_normalize",
    )(x)

    model = tf.keras.Model(inputs=inp, outputs=out)

    # Load backbone + pooling weights; ArcFace / species head weights are skipped.
    model.load_weights(str(weights_path), by_name=True, skip_mismatch=True)
    print(f"[model] Competition weights loaded from {weights_path}")

    return model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_model() -> tf.keras.Model:
    """Download weights if needed, build the embedding model, and cache it."""
    global _embedding_model, EMBEDDING_DIM

    if _embedding_model is not None:
        return _embedding_model

    _download_model_if_needed()

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model weights not found at {MODEL_PATH}.\n"
            "Set MODEL_PATH in .env to an existing file, or ensure network "
            "access so the app can download from HuggingFace."
        )

    print(f"[model] Loading {MODEL_PATH} …")
    _embedding_model = _build_and_load(MODEL_PATH)

    EMBEDDING_DIM = int(_embedding_model.output_shape[-1])
    print(f"[model] Ready.  embedding_dim={EMBEDDING_DIM}")

    return _embedding_model


def embed_image_array(
    model: tf.keras.Model,
    img_array: np.ndarray,
) -> list[float]:
    """
    Run inference on a pre-processed (1, 600, 600, 3) float32 array.
    Returns an L2-normalised embedding as a Python list of floats.
    """
    prediction = model(img_array, training=False).numpy()   # (1, 2560)
    norm = np.linalg.norm(prediction, axis=1, keepdims=True)
    return (prediction / np.maximum(norm, 1e-10))[0].tolist()
