"""Evaluate the HiEnWrite regression models on the held-out TEST split (run once, after training).

For each model and for the train-mean baseline: per-trait MAE, RMSE, R^2 and Pearson r on the test split,
macro (unweighted) means over traits, per-language breakdown, and participant-group bootstrap 95% intervals,
including the paired RMSE difference against the baseline. Predictions go through the same ``predict_gray``
path the Flask app uses. Writes JSON, per-model prediction CSVs and PNG plots.

Usage:
    python -m src.evaluation.evaluate_regression
"""
import argparse
import csv
import hashlib
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import config
from src.data.hienwrite import LANGUAGES, TARGETS, read_split_manifest, stack_targets, target_stats
from src.errors import ModelUnavailableError
from src.evaluation.metrics import regression_metrics
from src.preprocessing.image_processor import decode_image_file

BASELINE = 'mean_baseline'
N_BOOTSTRAP = 2000
# Reference palette (dataviz skill): neutral gray for the baseline, categorical slots 1-3 for models.
COLORS = {BASELINE: '#9a9993', 'hienwrite_resnet18': '#2a78d6', 'hienwrite_svr': '#eb6834', 'hienwrite_rf': '#1baf7a'}
INK, INK_2, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def load_models(models_dir=config.HIENWRITE_MODELS_DIR):
    from src.classical.predict_classical import ClassicalModel
    from src.deep_learning.predict_resnet18 import ResNet18Model

    models_dir = Path(models_dir)
    candidates = [('hienwrite_resnet18', ResNet18Model, models_dir / config.HIENWRITE_RESNET18_PATH.name),
                  ('hienwrite_svr', ClassicalModel, models_dir / config.HIENWRITE_SVR_PATH.name),
                  ('hienwrite_rf', ClassicalModel, models_dir / config.HIENWRITE_RF_PATH.name)]
    models, skipped = {}, {}
    for name, cls, path in candidates:
        try:
            model = cls(path)
        except ModelUnavailableError as exc:
            skipped[name] = str(exc)
            continue
        if model.metadata.get('label_mode') != 'regression' or list(model.label_names) != TARGETS:
            skipped[name] = 'Not a HiEnWrite five-target regression artifact.'
            continue
        model.artifact_path = path
        models[name] = model
    return models, skipped


def predict_split(model, samples):
    preds, latencies = [], []
    for s in samples:
        gray = decode_image_file(s['path'], min_side=1).gray
        started = time.perf_counter()
        out = model.predict_gray(gray)
        latencies.append((time.perf_counter() - started) * 1000)
        preds.append([out['scores'][t] for t in TARGETS])
    return np.asarray(preds, dtype=float), np.asarray(latencies)


def _rmse_per_trait(Y, P, M):
    out = np.full(len(TARGETS), np.nan)
    for i in range(len(TARGETS)):
        v = M[:, i]
        if v.any():
            out[i] = np.sqrt(np.mean((P[v, i] - Y[v, i]) ** 2))
    return out


def _r_per_trait(Y, P, M):
    out = np.full(len(TARGETS), np.nan)
    for i in range(len(TARGETS)):
        v = M[:, i]
        t, p = Y[v, i], P[v, i]
        if len(t) >= 3 and np.ptp(t) > 0 and np.ptp(p) > 0:
            out[i] = np.corrcoef(t, p)[0, 1]
    return out


def bootstrap(Y, M, groups, predictions, baseline, n=N_BOOTSTRAP, seed=config.SEED):
    """Resample participant groups with replacement (images of one group stay together). Returns, per model,
    95% percentile intervals of RMSE and Pearson r per trait and macro, and of RMSE(model) - RMSE(baseline)."""
    rng = np.random.default_rng(seed)
    unique = np.array(sorted(set(groups)))
    members = {g: np.nonzero(groups == g)[0] for g in unique}
    draws = {name: {'rmse': [], 'r': [], 'delta': []} for name in predictions}
    for _ in range(n):
        idx = np.concatenate([members[g] for g in rng.choice(unique, size=len(unique), replace=True)])
        y, m = Y[idx], M[idx]
        base = _rmse_per_trait(y, baseline[idx], m)
        for name, P in predictions.items():
            rmse = _rmse_per_trait(y, P[idx], m)
            draws[name]['rmse'].append(rmse)
            draws[name]['r'].append(_r_per_trait(y, P[idx], m))
            draws[name]['delta'].append(rmse - base)

    def interval(values):
        values = values[np.isfinite(values)]
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))] if len(values) else None

    out = {}
    for name, d in draws.items():
        rmse, r, delta = (np.array(d[k]) for k in ('rmse', 'r', 'delta'))
        with warnings.catch_warnings():  # a resample with no valid value for a trait gives NaN, excluded below
            warnings.simplefilter('ignore', RuntimeWarning)
            macro_rmse, macro_r, macro_delta = (np.nanmean(v, axis=1) for v in (rmse, r, delta))
        entry = {}
        for i, t in enumerate(TARGETS):
            entry[t] = {'rmse_95ci': interval(rmse[:, i]), 'pearson_r_95ci': interval(r[:, i]),
                        'rmse_minus_baseline_95ci': interval(delta[:, i]),
                        'share_of_resamples_better_than_baseline': float(np.mean(delta[:, i] < 0))}
        entry['macro'] = {'rmse_95ci': interval(macro_rmse), 'pearson_r_95ci': interval(macro_r),
                          'rmse_minus_baseline_95ci': interval(macro_delta),
                          'share_of_resamples_better_than_baseline': float(np.mean(macro_delta < 0))}
        out[name] = entry
    return {'method': f'participant-group bootstrap, {n} resamples, percentile 95% intervals, seed {seed}',
            'models': out}


def write_predictions(path, samples, P):
    Y, M = stack_targets(samples)
    with Path(path).open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'group', 'language', *[f'true_{t}' for t in TARGETS], *[f'valid_{t}' for t in TARGETS],
                         *[f'pred_{t}' for t in TARGETS]])
        for s, y, m, p in zip(samples, Y, M, P):
            writer.writerow([s['image'], s['group'], s['language'], *['' if np.isnan(v) else int(v) for v in y],
                             *[int(v) for v in m], *[round(float(v), 4) for v in p]])


def _style(ax):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save_plots(report, predictions, Y, M, out_dir, history_path, display_names):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    plt.rcParams.update({'font.size': 9, 'text.color': INK, 'axes.labelcolor': INK_2, 'figure.facecolor': '#fcfcfb',
                         'axes.facecolor': '#fcfcfb'})
    written = []
    names = [BASELINE, *report['models']]

    # Test RMSE per trait: baseline vs models, with group-bootstrap 95% intervals.
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    width = 0.8 / len(names)
    x = np.arange(len(TARGETS))
    for k, name in enumerate(names):
        entry = report['baseline'] if name == BASELINE else report['models'][name]
        rmse = [entry['test_metrics']['per_trait'][t]['rmse'] for t in TARGETS]
        cis = report['bootstrap']['models'].get(name, {})
        err = np.array([[r - cis[t]['rmse_95ci'][0], cis[t]['rmse_95ci'][1] - r] for t, r in zip(TARGETS, rmse)]).T \
            if cis else None  # the baseline is a fixed reference: no interval drawn
        ax.bar(x + (k - (len(names) - 1) / 2) * width, rmse, width * 0.9, color=COLORS.get(name, '#6250d6'),
               label=display_names.get(name, name), yerr=err, ecolor=INK_2, error_kw={'linewidth': 1, 'capsize': 2})
    ax.set_xticks(x, TARGETS)
    ax.set_ylabel('test RMSE (score points, lower is better)')
    ax.set_title(f"HiEnWrite test split (n = {report['n_test']} images): RMSE per trait with 95% bootstrap intervals",
                 fontsize=9, loc='left')
    ax.legend(frameon=False, fontsize=8, ncol=2)
    _style(ax)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / 'test_rmse_by_trait.png', dpi=130)
    plt.close(fig)
    written.append('test_rmse_by_trait.png')

    # Predicted vs true, one row per model.
    for name, P in predictions.items():
        fig, axes = plt.subplots(1, len(TARGETS), figsize=(13, 2.9))
        for i, (ax, t) in enumerate(zip(axes, TARGETS)):
            v = M[:, i]
            ax.scatter(Y[v, i], P[v, i], s=14, color=COLORS.get(name, '#6250d6'), alpha=0.75, linewidths=0)
            lo, hi = min(Y[v, i].min(), P[v, i].min()) - 1, max(Y[v, i].max(), P[v, i].max()) + 1
            ax.plot([lo, hi], [lo, hi], color=INK_2, linewidth=1, linestyle='--')
            m = report['models'][name]['test_metrics']['per_trait'][t]
            r = 'n/a' if m['pearson_r'] is None else f"{m['pearson_r']:+.2f}"
            ax.set_title(f"{t}\nRMSE {m['rmse']:.2f} · r {r} · n {m['n']}", fontsize=8, loc='left')
            ax.set_xlabel('self-reported score')
            if i == 0:
                ax.set_ylabel('predicted score')
            _style(ax)
        fig.suptitle(f'{display_names.get(name, name)}: test predictions (dashed = perfect)', fontsize=9, x=0.01, ha='left')
        fig.tight_layout()
        fig.savefig(Path(out_dir) / f'scatter_{name}.png', dpi=110)
        plt.close(fig)
        written.append(f'scatter_{name}.png')

    if history_path and Path(history_path).exists():
        history = json.loads(Path(history_path).read_text())
        epochs = [h['epoch'] for h in history['epochs']]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.2))
        a1.plot(epochs, [h['train_loss'] for h in history['epochs']], color=COLORS['hienwrite_resnet18'], linewidth=2,
                label='train loss')
        a1.plot(epochs, [h['val_loss'] for h in history['epochs']], color='#eb6834', linewidth=2, label='validation loss')
        a1.axvline(history['best_epoch'], color=INK_2, linewidth=1, linestyle=':')
        a1.text(history['best_epoch'], a1.get_ylim()[1], ' selected', color=INK_2, fontsize=8, va='top')
        a1.set_xlabel('epoch')
        a1.set_ylabel('masked MSE (standardised scores)')
        a1.legend(frameon=False, fontsize=8)
        a2.plot(epochs, [h['val_macro_rmse'] for h in history['epochs']], color='#eb6834', linewidth=2)
        a2.set_xlabel('epoch')
        a2.set_ylabel('validation macro RMSE (score points)')
        for ax in (a1, a2):
            _style(ax)
        fig.suptitle('HiEnWrite ResNet18 regression: training history', fontsize=9, x=0.01, ha='left')
        fig.tight_layout()
        fig.savefig(Path(out_dir) / 'resnet18_training_curves.png', dpi=130)
        plt.close(fig)
        written.append('resnet18_training_curves.png')
    return written


def evaluate(manifest=config.HIENWRITE_SPLITS_DIR / 'split_manifest.csv', out_dir=config.HIENWRITE_EVALUATION_DIR,
             models_dir=config.HIENWRITE_MODELS_DIR, data_root=None, n_bootstrap=N_BOOTSTRAP):
    samples, meta = read_split_manifest(manifest, data_root)
    train = [s for s in samples if s['split'] == 'train']
    test = [s for s in samples if s['split'] == 'test']
    if not test:
        raise ValueError('The split manifest has no test samples.')
    Y, M = stack_targets(test)
    groups = np.array([s['group'] for s in test])
    languages = np.array([s['language'] for s in test])
    stats = target_stats(train)
    baseline = np.tile([stats[t]['mean'] for t in TARGETS], (len(test), 1))

    def metrics_with_languages(P):
        result = {'test_metrics': regression_metrics(Y, P, TARGETS, M), 'by_language': {}}
        for code, label in LANGUAGES.items():
            sel = languages == code
            if sel.any():
                result['by_language'][label] = regression_metrics(Y[sel], P[sel], TARGETS, M[sel])
        return result

    models, skipped = load_models(models_dir)
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'dataset': 'HiEnWrite', 'task': 'five-target regression (raw questionnaire score scale)',
        'split_used': 'test', 'n_test': len(test), 'n_test_groups': int(len(set(groups))),
        'n_test_valid_per_trait': {t: int(M[:, i].sum()) for i, t in enumerate(TARGETS)},
        'manifest': str(Path(manifest).resolve()),
        'split': {k: meta[k] for k in ('dataset_fingerprint', 'seed', 'ratios', 'split_unit', 'n_samples', 'n_groups',
                                       'counts', 'target_distributions', 'masked_targets', 'csv_columns')},
        'aggregation': 'macro = unweighted mean over the five traits of each per-trait metric',
        'baseline': {'display_name': 'Train-mean baseline', 'description': 'predicts the training-split mean of each trait',
                     'train_means': {t: stats[t]['mean'] for t in TARGETS}, **metrics_with_languages(baseline)},
        'models': {}, 'skipped': skipped, 'warnings': [],
    }
    predictions = {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(out_dir / f'predictions_{BASELINE}.csv', test, baseline)
    for name, model in models.items():
        mmeta = model.metadata
        if mmeta.get('dataset_fingerprint') != meta['dataset_fingerprint']:
            report['warnings'].append(f'{name} was trained on a different split/dataset fingerprint; its test '
                                      'metrics may be contaminated. Retrain it on this manifest.')
        P, latency = predict_split(model, test)
        predictions[name] = P
        write_predictions(out_dir / f'predictions_{name}.csv', test, P)
        report['models'][name] = {
            'display_name': model.display_name,
            'artifact': Path(model.artifact_path).name, 'artifact_sha256_16': _sha256(model.artifact_path),
            'validation_metrics': mmeta.get('validation_metrics'),
            **metrics_with_languages(P),
            'latency_ms': {'mean': float(latency.mean()), 'median': float(np.median(latency)),
                           'note': 'model inference after decoding, single image'},
            'config': {k: mmeta.get(k) for k in ('estimator', 'feature_set', 'selected_params', 'preprocessing',
                                                 'pretrained', 'freeze_backbone', 'loss', 'best_epoch', 'epochs_run',
                                                 'learning_rate', 'batch_size', 'device', 'seed') if mmeta.get(k) is not None},
        }
    report['bootstrap'] = bootstrap(Y, M, groups, predictions, baseline, n_bootstrap)
    display_names = {BASELINE: 'Train-mean baseline', **{n: m.display_name for n, m in models.items()}}
    report['plots'] = save_plots(report, predictions, Y, M, out_dir, Path(models_dir) / config.HIENWRITE_RESNET18_HISTORY_PATH.name,
                                 display_names)
    stamp = report['created_utc'].replace(':', '').replace('-', '')
    text = json.dumps(report, indent=2)
    (out_dir / f'evaluation_{stamp}.json').write_text(text)
    (out_dir / 'latest.json').write_text(text)
    return report


def summary_lines(report):
    rows = [('Train-mean baseline', report['baseline'], None)]
    rows += [(e['display_name'], e, report['bootstrap']['models'][n]) for n, e in report['models'].items()]
    lines = [f"HiEnWrite test split: {report['n_test']} images, {report['n_test_groups']} participant groups; "
             f"valid per trait {report['n_test_valid_per_trait']}",
             f"{'model':50s} {'MAE':>6s} {'RMSE':>6s} {'R2':>7s} {'r':>7s}  RMSE-baseline 95% CI"]
    for label, e, boot in rows:
        m = e['test_metrics']['macro']
        r = '   n/a' if m['pearson_r'] is None else f"{m['pearson_r']:+.3f}"
        ci = boot['macro']['rmse_minus_baseline_95ci'] if boot else None
        lines.append(f"{label:50s} {m['mae']:6.3f} {m['rmse']:6.3f} {m['r2']:+7.3f} {r:>7s}  "
                     + (f'[{ci[0]:+.3f}, {ci[1]:+.3f}]' if ci else '-'))
    return lines


def main():
    parser = argparse.ArgumentParser(description='Evaluate HiEnWrite regression models on the held-out test split.')
    parser.add_argument('--manifest', default=str(config.HIENWRITE_SPLITS_DIR / 'split_manifest.csv'))
    parser.add_argument('--data-root', help='Dataset folder, if it moved since the manifest was written.')
    parser.add_argument('--models-dir', default=str(config.HIENWRITE_MODELS_DIR))
    parser.add_argument('--out-dir', default=str(config.HIENWRITE_EVALUATION_DIR))
    parser.add_argument('--bootstrap', type=int, default=N_BOOTSTRAP)
    args = parser.parse_args()
    try:
        report = evaluate(args.manifest, args.out_dir, args.models_dir, args.data_root, args.bootstrap)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f'error: {exc}')
    print('\n'.join(summary_lines(report)))
    for name, reason in report['skipped'].items():
        print(f'{name}: skipped ({reason})')
    for warning in report['warnings']:
        print(f'warning: {warning}')
    print(f"Saved {Path(args.out_dir) / 'latest.json'} and plots {report['plots']}")


if __name__ == '__main__':
    main()
