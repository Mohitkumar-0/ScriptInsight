"""End-to-end pipeline test: split -> train SVM/RF/ResNet18 -> load -> evaluate.

Uses a SYNTHETIC dataset with RANDOM labels in a temporary directory. It verifies that the code runs
and that artifacts are consistent; the resulting models are meaningless and are never saved in the
project or reported as results.
"""
import json

import numpy as np

from src.classical.predict_classical import ClassicalModel
from src.classical.train_classical import QUICK_RF_GRID, QUICK_SVM_GRID, save_bundle, train_model
from src.data.dataset import prepare, read_manifest
from src.deep_learning.predict_resnet18 import ResNet18Model
from src.deep_learning.train_resnet18 import train
from src.evaluation.evaluate import evaluate
from tests.conftest import synthetic_writer_image


def test_full_pipeline_on_synthetic_data(multilabel_dataset, tmp_path):
    manifest, meta = prepare(multilabel_dataset, tmp_path / 'splits')
    samples, meta = read_manifest(manifest)
    counts = {s: sum(x['split'] == s for x in samples) for s in ('train', 'val', 'test')}

    classical_dir, deep_dir = tmp_path / 'classical', tmp_path / 'deep'
    for name, grid in (('svm', QUICK_SVM_GRID), ('random_forest', QUICK_RF_GRID)):
        bundle = train_model(name, samples, meta, True, 0, grid, tmp_path / 'cache')
        assert (bundle['n_train'], bundle['n_val']) == (counts['train'], counts['val'])  # test split unused
        assert len(bundle['estimators']) == 5  # one binary classifier per trait
        save_bundle(bundle, classical_dir / f'{name}.joblib')
        model = ClassicalModel(classical_dir / f'{name}.joblib')
        out = model.predict_gray(synthetic_writer_image(999))
        assert out['task'] == 'multilabel' and all(0 <= v <= 1 for v in out['scores'].values())
        assert (classical_dir / f'{name}.json').exists()

    checkpoint, history = train(manifest, deep_dir, epochs=1, batch_size=16, pretrained=False, seed=0, device='cpu')
    assert len(history) == 1 and np.isfinite(history[0]['train_loss'])
    resnet = ResNet18Model(checkpoint)
    assert resnet.metadata['n_train'] == counts['train'] and resnet.metadata['pretrained'] is None
    assert json.loads((deep_dir / 'resnet18_history.json').read_text())['best_epoch'] == 1

    report = evaluate(manifest, tmp_path / 'eval', classical_dir, deep_dir)
    assert set(report['models']) == {'svm', 'random_forest', 'resnet18'}
    assert report['n_test'] == counts['test']
    for entry in report['models'].values():
        per_trait = entry['test_metrics']['per_trait']
        assert list(per_trait) == meta['label_names']
        assert all(np.array(m['confusion_matrix']).shape == (2, 2) for m in per_trait.values())
        assert sum(np.array(per_trait['Openness']['confusion_matrix']).ravel()) == counts['test']
        assert entry['validation_accuracy_target']['applicable'] is True
    assert (tmp_path / 'eval' / 'latest.json').exists()
    assert 'resnet18_loss.png' in report['plots']
