"""Nested dataset layout (<root>/training_set/<Trait>, <root>/test_set/<Trait>): merged, then split by source photo."""
import cv2
import numpy as np

import config
from src.data.dataset import load_samples, prepare, read_manifest


def _write_image(path, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cv2.imwrite(str(path), rng.integers(0, 255, (160, 160), dtype=np.uint8))


def test_nested_layout_is_merged_and_variants_stay_together(tmp_path):
    root = tmp_path / 'dataset'
    seed = 0
    for split in ('training_set', 'test_set'):
        for trait in config.TRAITS:
            for photo in range(3):
                for variant in range(1, 5):
                    _write_image(root / split / trait / f'IMG_{split[:2]}{trait[:3]}{photo}-{variant}.jpg', seed)
                    seed += 1

    samples, mode, names = load_samples(root)
    assert mode == 'multiclass'
    assert names == sorted(config.TRAITS)
    assert len(samples) == 2 * len(config.TRAITS) * 3 * 4

    manifest, meta = prepare(root, tmp_path / 'out', seed=1, group_regex=r'^(.+?)-\d', notes=['demo only'])
    assert 'demo only' in meta['warnings']
    assert meta['cross_split_groups'] == 0
    assert meta['n_groups'] == 2 * len(config.TRAITS) * 3

    loaded, _ = read_manifest(manifest)
    splits_per_group = {}
    for s in loaded:
        splits_per_group.setdefault(s['group'], set()).add(s['split'])
    assert all(len(v) == 1 for v in splits_per_group.values())
