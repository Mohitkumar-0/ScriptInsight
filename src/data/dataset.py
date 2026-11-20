"""Dataset discovery, leakage-aware 80:10:10 splitting and manifest handling.

Two dataset layouts are supported (see docs/DATASET.md):

* multilabel  - ``<root>/labels.csv`` with columns ``image``, optional ``participant_id`` and one 0/1
                column per Big Five trait (1 = high pole). This is what the report describes
                (five independent trait labels from a validated questionnaire).
* multiclass  - ``<root>/<Trait>/*.jpg`` folders, one mutually exclusive category per image. This only
                supports "which of five categories", NOT five independent trait predictions.

Splitting happens on groups (participant, or near-duplicate cluster when no participant id exists),
before any augmentation, so the same writer or the same source image never lands in two splits.

Usage:
    python -m src.data.dataset --data-root data/hwxpi --out artifacts/splits
"""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

import config
from src.preprocessing.image_processor import ImageValidationError, decode_image_file

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff', '.gif'}
N_FOLDS = 10  # one fold test, one fold validation, eight folds train -> 80:10:10


def load_samples(data_root):
    """Return (samples, label_mode, label_names). Each sample: dict(path, participant_id, labels)."""
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f'Dataset directory not found: {root}')

    labels_csv = root / 'labels.csv'
    if labels_csv.exists():
        samples = []
        with labels_csv.open(newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            missing = [t for t in config.TRAITS if t not in (reader.fieldnames or [])]
            if 'image' not in (reader.fieldnames or []) or missing:
                raise ValueError(f'labels.csv needs an "image" column and trait columns; missing: {missing}')
            for row in reader:
                labels = []
                for trait in config.TRAITS:
                    value = row[trait].strip()
                    if value not in {'0', '1'}:
                        raise ValueError(f'Trait labels must be 0/1; got {value!r} for {trait} in {row["image"]}')
                    labels.append(int(value))
                samples.append({'path': str((root / row['image']).resolve()),
                                'participant_id': (row.get('participant_id') or '').strip() or None,
                                'labels': labels})
        return samples, 'multilabel', list(config.TRAITS)

    class_dirs = sorted(d for d in root.iterdir() if d.is_dir() and d.name in config.TRAITS)
    if not class_dirs:
        # Nested layout such as <root>/training_set/<Trait>/ and <root>/test_set/<Trait>/ (Kaggle handwriting set).
        # Every trait folder below the root is merged; the supplied train/test division is NOT trusted
        # (writers and source photos overlap), so a new group-aware split is made afterwards.
        nested = sorted(d for d in root.rglob('*') if d.is_dir() and d.name in config.TRAITS)
        if not nested:
            raise ValueError(f'No labels.csv and no trait-named folders found in {root}')
        names = sorted({d.name for d in nested})
        samples = [{'path': str(p.resolve()), 'participant_id': None, 'labels': names.index(d.name)}
                   for d in nested for p in sorted(d.rglob('*')) if p.suffix.lower() in IMAGE_SUFFIXES]
        return samples, 'multiclass', names
    names = [d.name for d in class_dirs]
    samples = [{'path': str(p.resolve()), 'participant_id': None, 'labels': idx}
               for idx, d in enumerate(class_dirs)
               for p in sorted(d.rglob('*')) if p.suffix.lower() in IMAGE_SUFFIXES]
    return samples, 'multiclass', names


def _dhash(gray, size=16):
    import cv2
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).flatten()


def assign_groups(samples, group_regex=None, near_duplicate_bits=6):
    """Group samples so related images stay in one split. Returns the grouping method used."""
    if all(s['participant_id'] for s in samples):
        for s in samples:
            s['group'] = f"p:{s['participant_id']}"
        return 'participant_id'

    if group_regex:
        pattern = re.compile(group_regex)
        for s in samples:
            match = pattern.search(Path(s['path']).name)
            s['group'] = f"r:{match.group(1)}" if match else f"f:{Path(s['path']).name}"
        return f'filename_regex:{group_regex}'

    # Best effort for pre-augmented datasets without ids: cluster near-identical images (dHash).
    hashes = []
    for s in samples:
        try:
            hashes.append(_dhash(decode_image_file(s['path'], min_side=1).gray))
        except ImageValidationError:
            hashes.append(np.zeros(16 * 16, dtype=bool))
    hashes = np.packbits(np.array(hashes), axis=1)
    parent = list(range(len(samples)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    popcount = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1)
    for i in range(len(samples)):
        distances = popcount[np.bitwise_xor(hashes[i + 1:], hashes[i])].sum(1)
        for j in np.nonzero(distances <= near_duplicate_bits)[0] + i + 1:
            parent[find(int(j))] = find(i)
    for i, s in enumerate(samples):
        s['group'] = f'dup:{find(i)}'
    return f'near_duplicate_dhash<={near_duplicate_bits}bits'


def split_samples(samples, label_mode, seed=config.SEED):
    """Group-aware, stratified-where-possible 80:10:10 split. Returns a list of warnings."""
    warnings = []
    groups = np.array([s['group'] for s in samples])
    if label_mode == 'multilabel':
        strata = np.array([''.join(map(str, s['labels'])) for s in samples])
    else:
        strata = np.array([str(s['labels']) for s in samples])

    n_groups = len(set(groups))
    if n_groups < N_FOLDS:
        raise ValueError(f'Need at least {N_FOLDS} independent groups to make an 80:10:10 split; found {n_groups}.')

    counts = Counter(strata)
    if min(counts.values()) >= N_FOLDS:
        folds = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        fold_iter = folds.split(np.zeros(len(samples)), strata, groups)
    else:
        warnings.append('Some label combinations have fewer than 10 samples; split is grouped but not '
                        'stratified.')
        rng = np.random.default_rng(seed)
        unique = np.array(sorted(set(groups)))
        shuffled = {g: i for i, g in enumerate(rng.permutation(unique))}
        fold_iter = GroupKFold(n_splits=N_FOLDS).split(
            np.zeros(len(samples)), strata, np.array([shuffled[g] for g in groups]))

    fold_of = np.empty(len(samples), dtype=int)
    for fold, (_, held) in enumerate(fold_iter):
        fold_of[held] = fold
    for s, fold in zip(samples, fold_of):
        s['split'] = 'test' if fold == 0 else 'val' if fold == 1 else 'train'
    return warnings


def dataset_fingerprint(samples):
    digest = hashlib.sha256()
    for s in sorted(samples, key=lambda s: s['path']):
        digest.update(f"{Path(s['path']).name}:{Path(s['path']).stat().st_size}".encode())
    return digest.hexdigest()[:16]


def label_distribution(samples, label_mode, label_names):
    out = {}
    for split in ('train', 'val', 'test'):
        subset = [s for s in samples if s['split'] == split]
        if label_mode == 'multilabel':
            out[split] = {'n': len(subset), 'high_pole_count': {
                name: int(sum(s['labels'][i] for s in subset)) for i, name in enumerate(label_names)}}
        else:
            out[split] = {'n': len(subset), 'class_count': {
                name: int(sum(s['labels'] == i for s in subset)) for i, name in enumerate(label_names)}}
    return out


def prepare(data_root, out_dir, seed=config.SEED, group_regex=None, notes=None):
    samples, label_mode, label_names = load_samples(data_root)
    if not samples:
        raise ValueError(f'No images found in {data_root}')
    grouping = assign_groups(samples, group_regex)
    warnings = split_samples(samples, label_mode, seed)
    if label_mode == 'multiclass':
        warnings.append('Dataset has ONE category per image. Models trained on it choose one of five '
                        'categories; they do not predict five independent Big Five trait levels.')
    warnings.extend(notes or [])

    leaks = {g for g in {s['group'] for s in samples}
             if len({s['split'] for s in samples if s['group'] == g}) > 1}
    if leaks:
        raise RuntimeError(f'Leakage: {len(leaks)} groups appear in more than one split.')

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / 'manifest.csv'
    with manifest.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'group', 'split', 'labels'])
        for s in samples:
            labels = ' '.join(map(str, s['labels'])) if label_mode == 'multilabel' else str(s['labels'])
            writer.writerow([s['path'], s['group'], s['split'], labels])

    meta = {
        'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'data_root': str(Path(data_root).resolve()),
        'dataset_fingerprint': dataset_fingerprint(samples),
        'label_mode': label_mode,
        'label_names': label_names,
        'seed': seed,
        'ratios': {'train': 0.8, 'val': 0.1, 'test': 0.1},
        'grouping': grouping,
        'n_samples': len(samples),
        'n_groups': len({s['group'] for s in samples}),
        'distribution': label_distribution(samples, label_mode, label_names),
        'cross_split_groups': 0,
        'warnings': warnings,
    }
    (out / 'split_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return manifest, meta


def read_manifest(manifest_path):
    """Return (samples, meta) from a manifest written by prepare()."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists() or not (manifest_path.parent / 'split_meta.json').exists():
        raise FileNotFoundError(f'Split manifest not found at {manifest_path}. Create it first with: '
                                'python -m src.data.dataset --data-root <dataset folder>')
    meta = json.loads((manifest_path.parent / 'split_meta.json').read_text(encoding='utf-8'))
    samples = []
    with manifest_path.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            labels = ([int(v) for v in row['labels'].split()] if meta['label_mode'] == 'multilabel'
                      else int(row['labels']))
            samples.append({'path': row['path'], 'group': row['group'], 'split': row['split'], 'labels': labels})
    return samples, meta


def main():
    parser = argparse.ArgumentParser(description='Create a leakage-aware 80:10:10 split manifest.')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--out', default=str(config.SPLITS_DIR))
    parser.add_argument('--seed', type=int, default=config.SEED)
    parser.add_argument('--group-regex', help='Regex with one capture group giving the original-sample id '
                                              'from a filename (used when there is no participant_id).')
    parser.add_argument('--note', action='append', default=[],
                        help='Extra warning stored in the split metadata and shown with the results (repeatable).')
    args = parser.parse_args()
    try:
        manifest, meta = prepare(args.data_root, args.out, args.seed, args.group_regex, args.note)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f'error: {exc}\nSee docs/DATASET.md for the expected dataset layout.')
    print(f'Manifest: {manifest}')
    print(json.dumps({k: meta[k] for k in ('label_mode', 'n_samples', 'n_groups', 'grouping', 'warnings')}, indent=2))
    print(json.dumps(meta['distribution'], indent=2))


if __name__ == '__main__':
    main()
