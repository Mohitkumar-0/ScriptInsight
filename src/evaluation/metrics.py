"""Metric definitions shared by training (model selection) and evaluation.

multiclass : mutually exclusive categories. Accuracy, macro and weighted precision/recall/F1,
             per-class metrics and a KxK confusion matrix (rows = true, columns = predicted,
             both in ``label_names`` order).
multilabel : five independent binary traits. Per-trait accuracy/precision/recall/F1 of the high pole
             at a fixed 0.5 probability threshold, per-trait 2x2 confusion matrix
             [[TN, FP], [FN, TP]], macro averages over traits and exact-match (subset) accuracy.
"""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

THRESHOLD = 0.5


def multiclass_metrics(y_true, y_pred, label_names):
    labels = list(range(len(label_names)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    out = {
        'task': 'multiclass',
        'n': int(len(y_true)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'chance_accuracy_majority': float(np.bincount(y_true, minlength=len(labels)).max() / max(len(y_true), 1)),
        'per_class': {name: {'precision': float(p[i]), 'recall': float(r[i]), 'f1': float(f[i]), 'support': int(s[i])}
                      for i, name in enumerate(label_names)},
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        'label_order': list(label_names),
    }
    for average in ('macro', 'weighted'):
        p_avg, r_avg, f_avg, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=average, zero_division=0)
        out[average] = {'precision': float(p_avg), 'recall': float(r_avg), 'f1': float(f_avg)}
    return out


def multilabel_metrics(y_true, probabilities, label_names, threshold=THRESHOLD):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = (np.asarray(probabilities) >= threshold).astype(int)
    per_trait = {}
    for i, name in enumerate(label_names):
        t, pr = y_true[:, i], y_pred[:, i]
        p, r, f, _ = precision_recall_fscore_support(t, pr, labels=[1], average=None, zero_division=0)
        per_trait[name] = {
            'accuracy': float(accuracy_score(t, pr)),
            'precision': float(p[0]), 'recall': float(r[0]), 'f1': float(f[0]),
            'support_high': int(t.sum()), 'support_low': int(len(t) - t.sum()),
            'majority_baseline_accuracy': float(max(t.mean(), 1 - t.mean())) if len(t) else 0.0,
            'confusion_matrix': confusion_matrix(t, pr, labels=[0, 1]).tolist(),
        }
    return {
        'task': 'multilabel',
        'n': int(len(y_true)),
        'threshold': threshold,
        'per_trait': per_trait,
        'macro': {k: float(np.mean([m[k] for m in per_trait.values()])) for k in ('accuracy', 'precision', 'recall', 'f1')},
        'subset_accuracy': float(np.mean(np.all(y_true == y_pred, axis=1))) if len(y_true) else 0.0,
        'label_order': list(label_names),
    }


def selection_score(metrics):
    """Single number used for model selection on the validation split (never on test)."""
    return metrics['macro']['f1']


def accuracy_target_check(metrics, traits, target):
    """Report-specified target (e.g. >= 85% for Openness and Conscientiousness). Only meaningful for
    multilabel models, where each trait has its own accuracy."""
    if metrics.get('task') != 'multilabel':
        return {'applicable': False,
                'reason': 'Per-trait accuracy needs independent trait labels (multilabel task).'}
    results = {t: {'accuracy': metrics['per_trait'][t]['accuracy'],
                   'met': metrics['per_trait'][t]['accuracy'] >= target} for t in traits}
    return {'applicable': True, 'target': target, 'traits': results,
            'all_met': all(r['met'] for r in results.values())}
