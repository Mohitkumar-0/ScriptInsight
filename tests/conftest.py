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
