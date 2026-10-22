import numpy as np

from src.evaluation import metrics


def test_multiclass_metrics_hand_computed():
    m = metrics.multiclass_metrics(np.array([0, 0, 1, 2]), np.array([0, 1, 1, 2]), ['A', 'B', 'C'])
    assert m['accuracy'] == 0.75
    assert m['confusion_matrix'] == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]  # rows true, cols predicted
    assert m['label_order'] == ['A', 'B', 'C']
    assert m['per_class']['B'] == {'precision': 0.5, 'recall': 1.0, 'f1': 2 / 3, 'support': 1}
    assert np.isclose(m['macro']['f1'], (2 / 3 + 2 / 3 + 1) / 3)


def test_multiclass_confusion_matrix_keeps_absent_classes():
    m = metrics.multiclass_metrics(np.array([0, 0]), np.array([0, 0]), ['A', 'B', 'C', 'D', 'E'])
    assert np.array(m['confusion_matrix']).shape == (5, 5)


def test_multilabel_metrics_per_trait():
    y = np.array([[1, 0], [0, 0], [1, 1], [0, 1]])
    p = np.array([[0.9, 0.2], [0.6, 0.1], [0.4, 0.8], [0.1, 0.7]])  # threshold 0.5
    m = metrics.multilabel_metrics(y, p, ['T1', 'T2'])
    t1 = m['per_trait']['T1']
    assert t1['confusion_matrix'] == [[1, 1], [1, 1]] and t1['accuracy'] == 0.5
    assert m['per_trait']['T2']['accuracy'] == 1.0
    assert m['macro']['accuracy'] == 0.75 and m['subset_accuracy'] == 0.5


def test_accuracy_target_only_applies_to_multilabel():
    mc = metrics.multiclass_metrics(np.array([0, 1]), np.array([0, 1]), ['A', 'B'])
    assert metrics.accuracy_target_check(mc, ['A'], 0.85)['applicable'] is False
    ml = metrics.multilabel_metrics(np.array([[1, 0]] * 10), np.array([[0.9, 0.1]] * 9 + [[0.1, 0.1]]), ['O', 'C'])
    check = metrics.accuracy_target_check(ml, ['O', 'C'], 0.85)
    assert check['traits']['O'] == {'accuracy': 0.9, 'met': True} and check['all_met'] is True
