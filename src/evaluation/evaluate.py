"""Evaluate all trained models on the same held-out test split and save JSON + PNG artifacts.

Only models trained on the same manifest labels (same label_mode and label order) are compared.
The legacy Keras ResNet50 is excluded: it was trained on the original (augmented) copies of this same data with a
different split, so some of its training images are now in the test split and its numbers would not be comparable.

Usage:
    python -m src.evaluation.evaluate --manifest artifacts/splits/manifest.csv
"""
import argparse
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import config
from src.data.dataset import read_manifest
from src.errors import ModelUnavailableError
from src.evaluation.metrics import accuracy_target_check, multiclass_metrics, multilabel_metrics
from src.preprocessing.image_processor import decode_image_file


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def load_trained_models(classical_dir=config.CLASSICAL_DIR, deep_dir=config.DEEP_DIR):
    from src.classical.predict_classical import ClassicalModel
    from src.deep_learning.predict_resnet18 import ResNet18Model

    candidates = [('svm', ClassicalModel, Path(classical_dir) / 'svm.joblib'),
                  ('random_forest', ClassicalModel, Path(classical_dir) / 'random_forest.joblib'),
                  ('resnet18', ResNet18Model, Path(deep_dir) / 'resnet18_best.pt')]
    models, skipped = {}, {}
    for name, cls, path in candidates:
        try:
            model = cls(path)
            model.artifact_path = path
            models[name] = model
        except ModelUnavailableError as exc:
            skipped[name] = str(exc)
    return models, skipped


def evaluate_model(model, test_samples, label_mode, label_names):
    scores, latencies = [], []
    for s in test_samples:
        gray = decode_image_file(s['path'], min_side=1).gray
        started = time.perf_counter()
        out = model.predict_gray(gray)
        latencies.append(time.perf_counter() - started)
        scores.append([out['scores'][name] for name in label_names])
    scores = np.asarray(scores)
    if label_mode == 'multilabel':
        metrics = multilabel_metrics(np.array([s['labels'] for s in test_samples]), scores, label_names)
    else:
        metrics = multiclass_metrics(np.array([s['labels'] for s in test_samples]), scores.argmax(1), label_names)
    lat = np.asarray(latencies) * 1000
    metrics['latency_ms'] = {'mean': float(lat.mean()), 'median': float(np.median(lat)),
                             'p95': float(np.percentile(lat, 95)), 'n': int(len(lat)),
                             'note': 'model inference only (after decoding), single image, CPU'}
    return metrics


def save_plots(report, out_dir, history_path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    written = []
    for name, entry in report['models'].items():
        m = entry['test_metrics']
        mats = ([(name, m['confusion_matrix'], m['label_order'])] if m['task'] == 'multiclass'
                else [(t, v['confusion_matrix'], ['low', 'high']) for t, v in m['per_trait'].items()])
        fig, axes = plt.subplots(1, len(mats), figsize=(3.2 * len(mats) + 1, 3.4), squeeze=False)
        for ax, (title, cm, labels) in zip(axes[0], mats):
            cm = np.asarray(cm)
            ax.imshow(cm, cmap='Blues')
            ax.set_xticks(range(len(labels)), labels, rotation=45, ha='right', fontsize=7)
            ax.set_yticks(range(len(labels)), labels, fontsize=7)
            ax.set_xlabel('Predicted', fontsize=8)
            ax.set_ylabel('True', fontsize=8)
            ax.set_title(title, fontsize=9)
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, int(v), ha='center', va='center', fontsize=8)
        fig.suptitle(f'{entry["display_name"]} - test confusion matrix', fontsize=10)
        fig.tight_layout()
        path = Path(out_dir) / f'confusion_{name}.png'
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path.name)
    if history_path and Path(history_path).exists():
        history = json.loads(Path(history_path).read_text(encoding='utf-8'))['epochs']
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.plot([h['epoch'] for h in history], [h['train_loss'] for h in history], label='train loss')
        ax.plot([h['epoch'] for h in history], [h['val_loss'] for h in history], label='validation loss')
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.set_title('ResNet18 training history')
        ax.legend()
        fig.tight_layout()
        fig.savefig(Path(out_dir) / 'resnet18_loss.png', dpi=120)
        plt.close(fig)
        written.append('resnet18_loss.png')
    return written


def evaluate(manifest, out_dir=config.EVALUATION_DIR, classical_dir=config.CLASSICAL_DIR, deep_dir=config.DEEP_DIR):
    samples, meta = read_manifest(manifest)
    label_mode, label_names = meta['label_mode'], meta['label_names']
    test_samples = [s for s in samples if s['split'] == 'test']
    if not test_samples:
        raise ValueError('The manifest has no test samples.')

    models, skipped = load_trained_models(classical_dir, deep_dir)
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'manifest': str(Path(manifest).resolve()),
        'dataset': {k: meta[k] for k in ('dataset_fingerprint', 'label_mode', 'label_names', 'seed', 'grouping',
                                         'n_samples', 'n_groups', 'distribution', 'warnings')},
        'split_used': 'test', 'n_test': len(test_samples),
        'models': {}, 'skipped': skipped, 'warnings': [],
    }
    if label_mode == 'multiclass':
        # What a model gets by always predicting the most common training class (no learning at all).
        majority = Counter(s['labels'] for s in samples if s['split'] == 'train').most_common(1)[0][0]
        base = multiclass_metrics(np.array([s['labels'] for s in test_samples]),
                                  np.full(len(test_samples), majority), label_names)
        base_acc = base.get('accuracy', base['macro'].get('accuracy'))
        report['baseline_majority_class'] = {'class': label_names[majority], 'accuracy': base_acc,
                                             'macro_f1': base['macro']['f1']}
        report['warnings'].append(
            f'Majority-class baseline: always predicting {label_names[majority]} scores accuracy {base_acc:.3f} and '
            f'macro-F1 {base["macro"]["f1"]:.3f} on this test split. A model that does not clearly beat this has '
            'learned nothing useful.')
    for name, model in models.items():
        mmeta = model.metadata
        if mmeta['label_mode'] != label_mode or list(mmeta['label_names']) != list(label_names):
            report['skipped'][name] = 'Trained on a different label definition than this manifest.'
            continue
        if mmeta.get('dataset_fingerprint') != meta['dataset_fingerprint']:
            report['warnings'].append(f'{name} was trained on a different dataset fingerprint; results may not be comparable.')
        test_metrics = evaluate_model(model, test_samples, label_mode, label_names)
        report['models'][name] = {
            'display_name': model.display_name,
            'artifact': Path(model.artifact_path).name, 'artifact_sha256_16': _sha256(model.artifact_path),
            'score_type': mmeta.get('score_type'),
            'validation_metrics': mmeta.get('validation_metrics'),
            'test_metrics': test_metrics,
            'validation_accuracy_target': accuracy_target_check(
                mmeta.get('validation_metrics') or {}, config.ACCURACY_TARGET_TRAITS, config.ACCURACY_TARGET),
            'config': {k: mmeta.get(k) for k in ('selected_params', 'pretrained', 'freeze_backbone', 'best_epoch',
                                                 'feature_version', 'denoise', 'preprocessing', 'seed')
                       if mmeta.get(k) is not None},
        }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report['plots'] = save_plots(report, out_dir, Path(deep_dir) / 'resnet18_history.json')
    stamp = report['created_utc'].replace(':', '').replace('-', '')
    text = json.dumps(report, indent=2)
    (out_dir / f'evaluation_{stamp}.json').write_text(text, encoding='utf-8')
    (out_dir / 'latest.json').write_text(text, encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models on the held-out test split.')
    parser.add_argument('--manifest', default=str(config.SPLITS_DIR / 'manifest.csv'))
    parser.add_argument('--out-dir', default=str(config.EVALUATION_DIR))
    args = parser.parse_args()
    try:
        report = evaluate(args.manifest, args.out_dir)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f'error: {exc}')
    for name, entry in report['models'].items():
        m = entry['test_metrics']
        print(f"{name:14s} test macro-F1 {m['macro']['f1']:.4f}  "
              f"accuracy {m.get('accuracy', m['macro'].get('accuracy')):.4f}  "
              f"median latency {m['latency_ms']['median']:.1f} ms")
    for message in report['warnings']:
        print(f'note: {message}')
    for name, reason in report['skipped'].items():
        print(f'{name:14s} skipped: {reason}')


if __name__ == '__main__':
    main()
