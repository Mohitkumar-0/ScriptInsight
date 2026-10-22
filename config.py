"""Central configuration for ScriptInsight.

Every value can be overridden with an environment variable of the same name
prefixed with ``SCRIPTINSIGHT_`` (for example ``SCRIPTINSIGHT_PORT=8000``).
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _env(name, default, cast=str):
    value = os.environ.get(f'SCRIPTINSIGHT_{name}')
    if value is None:
        return default
    if cast is bool:
        return value.strip().lower() in {'1', 'true', 'yes', 'on'}
    return cast(value)


APP_NAME = 'ScriptInsight'
APP_VERSION = '2.0.0'

# Server. Bind to localhost by default so the development server is not exposed to the network.
HOST = _env('HOST', '127.0.0.1')
PORT = _env('PORT', 5001, int)

# Canonical Big Five order (OCEAN), used by the dataset, training, API and dashboard.
TRAITS = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']

# Uploads
MAX_UPLOAD_BYTES = _env('MAX_UPLOAD_MB', 16, int) * 1024 * 1024
MIN_IMAGE_SIDE = _env('MIN_IMAGE_SIDE', 128, int)        # pixels, applies to width and height
MAX_IMAGE_PIXELS = 40_000_000                            # decompression-bomb guard
# File extensions accepted by the upload form; the decoded content must also be one of ALLOWED_FORMATS.
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp', 'bmp', 'tif', 'tiff', 'gif'}
ALLOWED_FORMATS = {'JPEG', 'PNG', 'WEBP', 'BMP', 'TIFF', 'GIF'}
UPLOAD_DIR = BASE_DIR / 'uploads'

# Preprocessing shared by the new pipelines (classical + ResNet18). Saved into each model's metadata.
IMAGE_SIZE = 224
DENOISE = _env('DENOISE', True, bool)                    # 3x3 Gaussian noise suppression
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Model artifacts
MODELS_DIR = BASE_DIR / 'models'
CLASSICAL_DIR = MODELS_DIR / 'classical'
DEEP_DIR = MODELS_DIR / 'deep_learning'
SVM_PATH = CLASSICAL_DIR / 'svm.joblib'
RF_PATH = CLASSICAL_DIR / 'random_forest.joblib'
RESNET18_PATH = DEEP_DIR / 'resnet18_best.pt'
RESNET18_HISTORY_PATH = DEEP_DIR / 'resnet18_history.json'

# Legacy Keras ResNet50 demo model (kept until a verified replacement is trained)
ENABLE_LEGACY_MODEL = _env('ENABLE_LEGACY_MODEL', True, bool)
LEGACY_MODEL_PATH = BASE_DIR / 'best_model.h5'
# Class order used when the legacy model was trained (alphabetical folder order in the notebook).
LEGACY_CLASSES = ['Agreeableness', 'Conscientiousness', 'Extraversion', 'Neuroticism', 'Openness']

# Evaluation outputs
ARTIFACTS_DIR = BASE_DIR / 'artifacts'
EVALUATION_DIR = ARTIFACTS_DIR / 'evaluation'
SPLITS_DIR = ARTIFACTS_DIR / 'splits'
LEGACY_RECORDS_PATH = ARTIFACTS_DIR / 'legacy_resnet50' / 'recorded_metrics.json'

# Report requirement NFR: >= 85% validation accuracy for these traits.
ACCURACY_TARGET = 0.85
ACCURACY_TARGET_TRAITS = ['Openness', 'Conscientiousness']

SEED = _env('SEED', 42, int)

DISCLAIMER = (
    'Experimental research prototype. Inferring personality from handwriting is not a validated '
    'psychological assessment. Do not use these results for hiring, clinical, educational or any '
    'other consequential decision.'
)
