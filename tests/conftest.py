import csv
import io
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

SAMPLE = ROOT / '1.1.jpg'


def image_bytes(fmt, size=(300, 300), mode='RGB', color=(255, 255, 255)):
    """Encode a small synthetic 'handwriting' image (a few dark strokes on white) in the given format."""
    image = Image.new(mode, size, color if mode != 'RGBA' else (255, 255, 255, 0))
    array = np.array(image)
    for i in range(5):
        cv2.line(array, (10 + 20 * i, 20), (size[0] - 20, 40 + 30 * i), (0, 0, 0, 255)[:array.shape[-1]], 3)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format=fmt)
    return buffer.getvalue()


def handwriting_like_image(size=(900, 600), seed=0, ruled=True):
    """RGB page of cursive-font words on slightly noisy, ruled off-white paper. It only has the stroke/line
    structure of handwriting (it is drawn with OpenCV's script font), which is what the validator measures."""
    rng = np.random.default_rng(seed)
    w, h = size
    img = np.clip(np.full((h, w, 3), (235, 238, 240), float) + rng.normal(0, 3, (h, w, 3)), 0, 255).astype(np.uint8)
    words = 'the quick brown fox jumps over the lazy dog while my friend writes notes'.split()
    for y in range(60, h - 30, 55):
        if ruled:
            cv2.line(img, (0, y + 8), (w, y + 8), (200, 180, 170), 1)
        x = 30
        while x < w - 200:
            word = words[rng.integers(len(words))]
            cv2.putText(img, word, (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.1 * (0.8 + rng.random() * 0.5),
                        (60, 40, 30), 2, cv2.LINE_AA)
            x += 40 + len(word) * 22
    return img


def encode(array, fmt='PNG'):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format=fmt)
    return buffer.getvalue()


def synthetic_writer_image(seed, size=320):
    """Distinct pseudo-handwriting per seed, used only to exercise the pipeline code."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 255, np.uint8)
    for _ in range(25):
        pts = rng.integers(10, size - 10, size=(4, 2))
        cv2.polylines(img, [pts.reshape(-1, 1, 2).astype(np.int32)], False, int(rng.integers(0, 80)), int(rng.integers(1, 4)))
    return img


@pytest.fixture
def multilabel_dataset(tmp_path):
    """60 synthetic writers x 2 images, five random 0/1 trait labels per writer (pipeline test only)."""
    root = tmp_path / 'ml_data'
    (root / 'images').mkdir(parents=True)
    rng = np.random.default_rng(0)
    with (root / 'labels.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'participant_id', *config.TRAITS])
        for p in range(60):
            labels = rng.integers(0, 2, size=5)
            for k in range(2):
                name = f'images/w{p:03d}_{k}.png'
                cv2.imwrite(str(root / name), synthetic_writer_image(p * 10 + k))
                writer.writerow([name, f'writer{p:03d}', *labels])
    return root


@pytest.fixture
def multiclass_dataset(tmp_path):
    """Folder-per-category layout where each original image has 2 near-identical copies (simulating a
    pre-augmented dataset without ids)."""
    root = tmp_path / 'mc_data'
    for c, trait in enumerate(config.LEGACY_CLASSES):
        (root / trait).mkdir(parents=True)
        for i in range(6):
            base = synthetic_writer_image(1000 + c * 100 + i)
            cv2.imwrite(str(root / trait / f'orig{i}_a.png'), base)
            cv2.imwrite(str(root / trait / f'orig{i}_b.png'), np.clip(base.astype(int) + 3, 0, 255).astype(np.uint8))
    return root


def _tensorflow_loads():
    try:
        import tensorflow  # noqa: F401
        return True
    except Exception:  # missing, or installed but unable to load its native libraries
        return False


# The legacy Keras model needs a working TensorFlow (Python <= 3.11 with tensorflow-macos 2.13 in this project).
LEGACY_AVAILABLE = config.LEGACY_MODEL_PATH.exists() and _tensorflow_loads()


def make_hienwrite_dataset(root, n_participants=24, seed=0):
    """Synthetic data in the HiEnWrite layout (data/merged_data.csv + images/), with the defects found in the
    real dataset: a missing file, an empty file, a truncated file, a tiny image, duplicate images with agreeing
    and with conflicting labels, an image in a sub-folder, and Openness constant (24) for every Hindi row.
    For pipeline tests only; never used as training data or results."""
    root = Path(root)
    (root / 'data').mkdir(parents=True)
    (root / 'images' / 'sub').mkdir(parents=True)
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_participants):
        e, a, c, n = (int(v) for v in rng.integers(5, 40, size=4))
        age, gender = str(18 + p % 5), 'Female' if p % 3 else 'Male'
        for lang in 'eh':
            name = f'{p}{lang}.jpeg'
            folder = root / 'images' / ('sub' if p == 5 and lang == 'e' else '')
            cv2.imwrite(str(folder / name), synthetic_writer_image(seed * 1000 + p * 2 + (lang == 'h')))
            rows.append([name, e, a, c, n, int(rng.integers(8, 40)) if lang == 'e' else 24, age, gender])
    images = root / 'images'
    (images / '1h.jpeg').unlink()                                   # missing file
    (images / '2e.jpeg').write_bytes(b'')                           # empty file
    (images / '3e.jpeg').write_bytes((images / '3e.jpeg').read_bytes()[:200])  # truncated
    cv2.imwrite(str(images / '4h.jpeg'), synthetic_writer_image(99, size=48))   # too small
    (images / '7e.jpeg').write_bytes((images / '6e.jpeg').read_bytes())         # conflicting duplicate
    (images / '8h.jpeg').write_bytes((images / '8e.jpeg').read_bytes())         # same participant, labels agree
    with (root / 'data' / 'merged_data.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Unnamed: 0', 'image', 'Extroversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism',
                         'Openness to Experience', 'Age', 'Gender'])
        for i, row in enumerate(rows):
            writer.writerow([i, *row])
    return root


@pytest.fixture
def hienwrite_root(tmp_path):
    return make_hienwrite_dataset(tmp_path / 'HiEnWrite')


@pytest.fixture
def hienwrite_split(hienwrite_root, tmp_path):
    """(split_manifest_path, meta) for the synthetic HiEnWrite dataset."""
    from src.data import hienwrite
    manifest, _ = hienwrite.validate(hienwrite_root, tmp_path / 'validation', n_jobs=1)
    return hienwrite.make_splits(manifest, tmp_path / 'splits', seed=3)


def make_regression_artifacts(out_dir):
    """Untrained HiEnWrite-style artifacts (random ResNet18 weights; regressors fitted on random features) for
    testing loading, inference plumbing and the API only. Returns {model_name: path}."""
    import joblib
    import torch
    from src.classical.train_regression import make_regressor
    from src.data.hienwrite import TARGET_COLUMNS, TARGETS
    from src.deep_learning.model import build_resnet18
    from src.features.handwriting_features import FEATURE_DIM, FEATURE_VERSION

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {t: {'n': 10, 'mean': 20.0, 'std': 5.0, 'min': 5.0, 'max': 38.0} for t in TARGETS}
    val = {'macro': {'rmse': 5.0, 'mae': 4.0, 'r2': -0.01, 'pearson_r': 0.0}}
    torch.manual_seed(0)
    resnet = out_dir / 'resnet18_regression.pt'
    torch.save({'state_dict': build_resnet18(5).state_dict(), 'metadata': {
        'model_name': 'hienwrite_resnet18', 'display_name': 'HiEnWrite ResNet18 regression · experimental',
        'arch': 'resnet18', 'num_outputs': 5, 'label_mode': 'regression', 'label_names': TARGETS,
        'csv_columns': TARGET_COLUMNS, 'score_type': 'regression_score', 'dataset': 'HiEnWrite', 'experimental': True,
        'target_mean': [20.0] * 5, 'target_std': [5.0] * 5, 'target_stats': stats,
        'preprocessing': {'size': 224, 'denoise': True, 'fit': 'pad'},
        'validation_metrics': {**val, 'macro': {**val['macro'], 'nrmse': 0.97}}}}, resnet)
    rng = np.random.default_rng(0)
    X, y = rng.random((12, FEATURE_DIM)).astype(np.float32), rng.integers(5, 40, size=12).astype(float)
    paths = {'hienwrite_resnet18': resnet}
    for name, kind, feature_set, params, nrmse in (
            ('hienwrite_svr', 'svr', 'hog', {'pca': 4, 'C': 1.0, 'epsilon': 0.1}, 0.95),
            ('hienwrite_rf', 'random_forest', 'hog_lbp', {'n_estimators': 5}, 0.99)):
        cols = slice(0, 6084) if feature_set == 'hog' else slice(0, FEATURE_DIM)
        joblib.dump({'model_name': name, 'display_name': name, 'label_mode': 'regression', 'task': 'regression',
                     'label_names': TARGETS, 'estimators': [make_regressor(kind, params, 0).fit(X[:, cols], y)] * 5,
                     'feature_set': feature_set, 'feature_version': FEATURE_VERSION, 'feature_dim': FEATURE_DIM,
                     'preprocessing': {'size': 224, 'denoise': True, 'fit': 'pad'}, 'denoise': True,
                     'score_type': 'regression_score', 'dataset': 'HiEnWrite', 'experimental': True,
                     'target_stats': stats, 'validation_metrics': {**val, 'macro': {**val['macro'], 'nrmse': nrmse}}},
                    out_dir / f'{name}.joblib')
        paths[name] = out_dir / f'{name}.joblib'
    return paths
