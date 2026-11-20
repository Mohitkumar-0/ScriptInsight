"""Train HiEnWrite five-target regressors on HOG / HOG+LBP features (experimental).

* svr           : HOG (6084 dims) -> StandardScaler -> optional PCA -> RBF SVR, targets standardised.
* random_forest : HOG + LBP (6262 dims) -> RandomForestRegressor.

One regressor per trait: each trait has its own set of valid samples (Openness is masked for Hindi rows)
and its own hyper-parameters. Scalers, PCA and target scaling are fitted on the training split only (inside
the sklearn pipelines). Hyper-parameters are chosen per trait by validation RMSE; the test split is never
read here.

Usage:
    python -m src.classical.train_regression                     # both models
    python -m src.classical.train_regression --models svr --quick
"""
import argparse
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import config
from src.classical.train_classical import save_bundle
from src.data.hienwrite import TARGET_COLUMNS, TARGETS, read_split_manifest, stack_targets, target_stats
from src.evaluation.metrics import normalized_rmse, regression_metrics
from src.features.handwriting_features import FEATURE_DIM, FEATURE_VERSION, features_from_gray, select_features
from src.preprocessing.image_processor import decode_image_file

PREPROCESSING = {'size': config.IMAGE_SIZE, 'denoise': True, 'fit': 'pad'}
MODELS = {
    'svr': {'model_name': 'hienwrite_svr', 'display_name': 'HiEnWrite SVR (HOG) · experimental',
            'feature_set': 'hog', 'path': config.HIENWRITE_SVR_PATH},
    'random_forest': {'model_name': 'hienwrite_rf', 'display_name': 'HiEnWrite Random Forest (HOG + LBP) · experimental',
                      'feature_set': 'hog_lbp', 'path': config.HIENWRITE_RF_PATH},
}
SVR_GRID = {'pca': [64, None], 'C': [0.3, 1.0, 3.0], 'epsilon': [0.1, 0.5]}
RF_GRID = {'n_estimators': [500], 'max_features': ['sqrt', 0.1], 'min_samples_leaf': [5, 20]}
QUICK_SVR_GRID = {'pca': [8], 'C': [1.0], 'epsilon': [0.1]}
QUICK_RF_GRID = {'n_estimators': [20], 'max_features': ['sqrt'], 'min_samples_leaf': [5]}


def _features_one(path, prep):
    return features_from_gray(decode_image_file(path, min_side=1).gray, denoise=prep['denoise'], fit=prep['fit'])


def extract_features(samples, prep=PREPROCESSING, cache_dir=config.ARTIFACTS_DIR / 'features', n_jobs=-1):
    """Full HOG+LBP matrix in ``samples`` order. Cached by image content hash, feature version and preprocessing
    (extraction has no fitted state, so computing it for every split leaks nothing)."""
    key = hashlib.sha256(json.dumps([FEATURE_VERSION, prep, [s['sha256'] for s in samples]]).encode()).hexdigest()[:16]
    cache = Path(cache_dir) / f'hienwrite_{key}.npz' if cache_dir else None
    if cache and cache.exists():
        return np.load(cache)['X']
    from joblib import Parallel, delayed
    X = np.stack(Parallel(n_jobs=n_jobs)(delayed(_features_one)(s['path'], prep) for s in samples)).astype(np.float32)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, X=X)
    return X


def _grid(grid):
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def make_regressor(kind, params, seed):
    if kind == 'svr':
        steps = [('scaler', StandardScaler())]
        steps.append(('pca', PCA(n_components=params['pca'], random_state=seed) if params['pca'] else 'passthrough'))
        steps.append(('svr', SVR(kernel='rbf', gamma='scale', C=params['C'], epsilon=params['epsilon'])))
        return TransformedTargetRegressor(regressor=Pipeline(steps), transformer=StandardScaler())
    return RandomForestRegressor(random_state=seed, n_jobs=-1, **params)


def train_model(kind, samples, meta, X_all, seed=config.SEED, grid=None, prep=PREPROCESSING):
    spec = MODELS[kind]
    grid = grid or (SVR_GRID if kind == 'svr' else RF_GRID)
    index = {s['image']: i for i, s in enumerate(samples)}
    train = [s for s in samples if s['split'] == 'train']
    val = [s for s in samples if s['split'] == 'val']
    X_tr = select_features(X_all[[index[s['image']] for s in train]], spec['feature_set'])
    X_val = select_features(X_all[[index[s['image']] for s in val]], spec['feature_set'])
    Y_tr, M_tr = stack_targets(train)
    Y_val, M_val = stack_targets(val)

    estimators, selected, val_pred, search = [], {}, np.full(Y_val.shape, np.nan), {}
    for i, trait in enumerate(TARGETS):
        tr, va = M_tr[:, i], M_val[:, i]
        best = None
        search[trait] = []
        for params in _grid(grid):
            if params.get('pca') and params['pca'] >= tr.sum():
                continue  # more PCA components than training samples for this trait
            model = make_regressor(kind, params, seed).fit(X_tr[tr], Y_tr[tr, i])
            pred = model.predict(X_val[va])
            rmse = float(np.sqrt(np.mean((pred - Y_val[va, i]) ** 2)))
            search[trait].append({'params': params, 'val_rmse': rmse})
            if best is None or rmse < best[0]:
                best = (rmse, params, model)
        if best is None:
            raise ValueError(f'No hyper-parameter setting is feasible for {trait} with {int(tr.sum())} training samples.')
        estimators.append(best[2])
        selected[trait] = best[1]
        val_pred[va, i] = best[2].predict(X_val[va])

    stats = target_stats(train)
    val_metrics = regression_metrics(Y_val, val_pred, TARGETS, M_val)
    val_metrics['macro']['nrmse'] = normalized_rmse(val_metrics, {t: stats[t]['std'] for t in TARGETS})
    return {
        'model_name': spec['model_name'], 'display_name': spec['display_name'],
        'dataset': 'HiEnWrite', 'experimental': True,
        'label_mode': 'regression', 'task': 'regression',
        'label_names': list(TARGETS), 'csv_columns': TARGET_COLUMNS,
        'score_type': 'regression_score',
        'estimators': estimators,
        'estimator': 'per-trait ' + ('RBF SVR (scaled features, optional PCA, standardised target)' if kind == 'svr'
                                     else 'RandomForestRegressor'),
        'feature_set': spec['feature_set'], 'feature_version': FEATURE_VERSION, 'feature_dim': FEATURE_DIM,
        'preprocessing': dict(prep), 'denoise': prep['denoise'],
        'selected_params': selected, 'grid': grid, 'search': search,
        'selection_metric': 'validation RMSE per trait',
        'validation_metrics': val_metrics,
        'target_stats': stats,
        'n_train': int(len(train)), 'n_val': int(len(val)),
        'n_train_per_trait': {t: int(M_tr[:, i].sum()) for i, t in enumerate(TARGETS)},
        'dataset_fingerprint': meta['dataset_fingerprint'], 'split_seed': meta['seed'], 'seed': seed,
        'sklearn_version': sklearn.__version__,
        'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }


def main():
    parser = argparse.ArgumentParser(description='Train HiEnWrite HOG/LBP regressors (SVR, Random Forest).')
    parser.add_argument('--manifest', default=str(config.HIENWRITE_SPLITS_DIR / 'split_manifest.csv'))
    parser.add_argument('--data-root', help='Dataset folder, if it moved since the manifest was written.')
    parser.add_argument('--out-dir', default=str(config.HIENWRITE_MODELS_DIR))
    parser.add_argument('--models', nargs='+', default=list(MODELS), choices=list(MODELS))
    parser.add_argument('--seed', type=int, default=config.SEED)
    parser.add_argument('--quick', action='store_true', help='Tiny grids (pipeline smoke test only).')
    parser.add_argument('--jobs', type=int, default=-1)
    args = parser.parse_args()
    try:
        samples, meta = read_split_manifest(args.manifest, args.data_root)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f'error: {exc}')
    X_all = extract_features(samples, n_jobs=args.jobs)
    print(f'features: {X_all.shape} ({FEATURE_VERSION}, preprocessing {PREPROCESSING})')
    for kind in args.models:
        grid = (QUICK_SVR_GRID if kind == 'svr' else QUICK_RF_GRID) if args.quick else None
        bundle = train_model(kind, samples, meta, X_all, args.seed, grid)
        path = Path(args.out_dir) / MODELS[kind]['path'].name
        save_bundle(bundle, path)
        m = bundle['validation_metrics']
        print(f"{kind}: saved {path}; validation macro RMSE {m['macro']['rmse']:.3f}, MAE {m['macro']['mae']:.3f}, "
              f"Pearson r {m['macro']['pearson_r']:.3f}, normalised RMSE {m['macro']['nrmse']:.3f}")
        for t, params in bundle['selected_params'].items():
            pt = m['per_trait'][t]
            print(f"   {t:18s} n_val={pt['n']:3d} RMSE {pt['rmse']:.3f} r {pt['pearson_r'] if pt['pearson_r'] is None else round(pt['pearson_r'], 3)}  {params}")


if __name__ == '__main__':
    main()
