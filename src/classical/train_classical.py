"""Train the HOG+LBP SVM and Random Forest models from a split manifest.

* Features are extracted with ``src.features.handwriting_features`` (versioned, fixed dimension).
* StandardScaler is fitted on the training split only (inside each sklearn Pipeline).
* Hyper-parameters are chosen on the validation split by macro-F1; the test split is never read here.
* multilabel datasets: one independent binary classifier per Big Five trait (one-vs-rest style).
  multiclass datasets: one classifier over the mutually exclusive categories.
* Class imbalance: class_weight='balanced' (SVM) / 'balanced_subsample' (Random Forest).

Usage:
    python -m src.classical.train_classical --manifest artifacts/splits/manifest.csv
"""
import argparse
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config
from src.data.dataset import read_manifest
from src.evaluation.metrics import multiclass_metrics, multilabel_metrics, selection_score
from src.features.handwriting_features import FEATURE_DIM, FEATURE_VERSION, features_from_gray
from src.preprocessing.image_processor import decode_image_file

SVM_GRID = {'C': [0.1, 1.0, 10.0], 'gamma': ['scale', 1e-4]}
RF_GRID = {'n_estimators': [300], 'max_depth': [None, 20], 'max_features': ['sqrt', 0.05]}
QUICK_SVM_GRID = {'C': [1.0], 'gamma': ['scale']}
QUICK_RF_GRID = {'n_estimators': [20], 'max_depth': [None], 'max_features': ['sqrt']}


def extract_split_features(samples, split, denoise, cache_dir=None, fingerprint=''):
    subset = [s for s in samples if s['split'] == split]
    cache = None
    if cache_dir:
        key = hashlib.sha256(f'{fingerprint}|{FEATURE_VERSION}|{denoise}|{split}|{len(subset)}'.encode()).hexdigest()[:16]
        cache = Path(cache_dir) / f'{split}_{key}.npz'
        if cache.exists():
            data = np.load(cache)
            return data['X'], data['y'], subset
    X = np.stack([features_from_gray(decode_image_file(s['path'], min_side=1).gray, denoise=denoise) for s in subset])
    y = np.array([s['labels'] for s in subset])
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, X=X, y=y)
    return X, y, subset


def _make(model_name, params, seed):
    if model_name == 'svm':
        clf = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=seed, **params)
    else:
        clf = RandomForestClassifier(class_weight='balanced_subsample', random_state=seed, n_jobs=-1, **params)
    return Pipeline([('scaler', StandardScaler()), ('clf', clf)])


def _grid(grid):
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def _fit_select(model_name, grid, X_tr, y_tr, X_val, y_val, seed, label_names, binary=False):
    """Fit every grid point on train, keep the best by validation macro-F1 (F1 of the high pole if binary)."""
    best = None
    for params in _grid(grid):
        pipe = _make(model_name, params, seed).fit(X_tr, y_tr)
        if binary:
            proba = pipe.predict_proba(X_val)[:, list(pipe.classes_).index(1)]
            score = multilabel_metrics(y_val[:, None], proba[:, None], ['trait'])['macro']['f1']
        else:
            score = selection_score(multiclass_metrics(y_val, pipe.predict(X_val), label_names))
        if best is None or score > best[0]:
            best = (score, params, pipe)
    return best


def train_model(model_name, samples, meta, denoise, seed, grid, cache_dir):
    X_tr, y_tr, _ = extract_split_features(samples, 'train', denoise, cache_dir, meta['dataset_fingerprint'])
    X_val, y_val, _ = extract_split_features(samples, 'val', denoise, cache_dir, meta['dataset_fingerprint'])
    label_names = meta['label_names']

    if meta['label_mode'] == 'multilabel':
        estimators, params, proba_val = [], {}, []
        for i, trait in enumerate(label_names):
            if len(set(y_tr[:, i])) < 2:
                raise ValueError(f'Training split has only one class for {trait}; cannot train a binary classifier.')
            _, best_params, pipe = _fit_select(model_name, grid, X_tr, y_tr[:, i], X_val, y_val[:, i],
                                               seed, [trait], binary=True)
            estimators.append(pipe)
            params[trait] = best_params
            proba_val.append(pipe.predict_proba(X_val)[:, list(pipe.classes_).index(1)])
        val_metrics = multilabel_metrics(y_val, np.stack(proba_val, axis=1), label_names)
    else:
        _, params, estimators = _fit_select(model_name, grid, X_tr, y_tr, X_val, y_val, seed, label_names)
        val_metrics = multiclass_metrics(y_val, estimators.predict(X_val), label_names)

    return {
        'model_name': model_name,
        'display_name': 'SVM (HOG + LBP)' if model_name == 'svm' else 'Random Forest (HOG + LBP)',
        'label_mode': meta['label_mode'],
        'label_names': label_names,
        'estimators': estimators,
        'score_type': 'platt_probability' if model_name == 'svm' else 'vote_fraction',
        'feature_version': FEATURE_VERSION,
        'feature_dim': FEATURE_DIM,
        'denoise': denoise,
        'selected_params': params,
        'selection_metric': 'validation macro-F1',
        'validation_metrics': val_metrics,
        'n_train': int(len(X_tr)), 'n_val': int(len(X_val)),
        'dataset_fingerprint': meta['dataset_fingerprint'],
        'seed': seed,
        'sklearn_version': sklearn.__version__,
        'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }


def save_bundle(bundle, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    sidecar = {k: v for k, v in bundle.items() if k != 'estimators'}
    path.with_suffix('.json').write_text(json.dumps(sidecar, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(description='Train HOG+LBP SVM and Random Forest models.')
    parser.add_argument('--manifest', default=str(config.SPLITS_DIR / 'manifest.csv'))
    parser.add_argument('--out-dir', default=str(config.CLASSICAL_DIR))
    parser.add_argument('--models', nargs='+', default=['svm', 'random_forest'], choices=['svm', 'random_forest'])
    parser.add_argument('--no-denoise', action='store_true')
    parser.add_argument('--seed', type=int, default=config.SEED)
    parser.add_argument('--quick', action='store_true', help='Tiny grids (for pipeline smoke tests only).')
    parser.add_argument('--cache-dir', default=str(config.ARTIFACTS_DIR / 'features'))
    args = parser.parse_args()

    try:
        samples, meta = read_manifest(args.manifest)
    except FileNotFoundError as exc:
        raise SystemExit(f'error: {exc}')
    for name in args.models:
        grid = (QUICK_SVM_GRID if args.quick else SVM_GRID) if name == 'svm' else (QUICK_RF_GRID if args.quick else RF_GRID)
        bundle = train_model(name, samples, meta, not args.no_denoise, args.seed, grid, args.cache_dir)
        path = Path(args.out_dir) / ('svm.joblib' if name == 'svm' else 'random_forest.joblib')
        save_bundle(bundle, path)
        print(f'{name}: saved {path}; validation macro-F1 = {bundle["validation_metrics"]["macro"]["f1"]:.4f}')


if __name__ == '__main__':
    main()
