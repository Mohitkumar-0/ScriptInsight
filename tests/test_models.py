import joblib
import numpy as np
import pytest
import torch

import config
from src.classical.predict_classical import ClassicalModel
from src.deep_learning.model import build_resnet18
from src.deep_learning.predict_resnet18 import ResNet18Model
from src.errors import ModelUnavailableError
from src.preprocessing.image_processor import decode_image_file
from tests.conftest import SAMPLE, synthetic_writer_image


def test_resnet18_architecture_and_head():
    model = build_resnet18(5)
    assert model.fc.in_features == 512 and model.fc.out_features == 5
    assert sum(p.numel() for p in model.parameters()) == 11_179_077  # ResNet18 backbone + 512x5 head
    model.eval()
    with torch.no_grad():
        assert model(torch.zeros(2, 3, 224, 224)).shape == (2, 5)


def test_freeze_backbone_trains_only_head():
    model = build_resnet18(5, freeze_backbone=True)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {'fc.weight', 'fc.bias'}


def test_missing_checkpoints_raise_explicit_errors(tmp_path):
    with pytest.raises(ModelUnavailableError, match='not found'):
        ResNet18Model(tmp_path / 'resnet18_best.pt')
    with pytest.raises(ModelUnavailableError, match='not found'):
        ClassicalModel(tmp_path / 'svm.joblib')


def test_incompatible_artifacts_rejected(tmp_path):
    torch.save({'state_dict': {}, 'metadata': {'arch': 'resnet50'}}, tmp_path / 'bad.pt')
    with pytest.raises(ModelUnavailableError, match='not a ResNet18'):
        ResNet18Model(tmp_path / 'bad.pt')
    joblib.dump({'feature_version': 'old', 'feature_dim': 10}, tmp_path / 'old.joblib')
    with pytest.raises(ModelUnavailableError, match='Retrain'):
        ClassicalModel(tmp_path / 'old.joblib')


def _checkpoint(tmp_path, label_mode, labels):
    """Untrained weights saved to a temp file, only to test loading and output semantics."""
    model = build_resnet18(len(labels))
    meta = {'arch': 'resnet18', 'num_outputs': len(labels), 'label_mode': label_mode, 'label_names': labels,
            'preprocessing': {'size': 224, 'denoise': True}, 'score_type': 'x'}
    torch.save({'state_dict': model.state_dict(), 'metadata': meta}, tmp_path / 'r18.pt')
    return ResNet18Model(tmp_path / 'r18.pt')


def test_resnet18_multilabel_outputs_are_independent_probabilities(tmp_path):
    model = _checkpoint(tmp_path, 'multilabel', config.TRAITS)
    out = model.predict_gray(synthetic_writer_image(5))
    assert list(out['scores']) == config.TRAITS
    assert all(0 <= v <= 1 for v in out['scores'].values())
    assert set(out['levels'].values()) <= {'high', 'low'}
    assert out == model.predict_gray(synthetic_writer_image(5))  # deterministic in eval mode
    assert not model.model.training


def test_resnet18_multiclass_outputs_sum_to_one(tmp_path):
    out = _checkpoint(tmp_path, 'multiclass', config.LEGACY_CLASSES).predict_gray(synthetic_writer_image(5))
    assert np.isclose(sum(out['scores'].values()), 1.0, atol=1e-5)
    assert out['predicted_label'] == max(out['scores'], key=out['scores'].get)


@pytest.mark.skipif(not config.LEGACY_MODEL_PATH.exists(), reason='best_model.h5 not present')
def test_legacy_model_loads_trained_weights_and_is_deterministic():
    from src.deep_learning.legacy_resnet50 import LegacyResNet50Model
    model = LegacyResNet50Model()
    assert model.metadata['parameters'] == 24_805_253
    gray = decode_image_file(SAMPLE).gray
    first, second = model.predict_gray(gray), model.predict_gray(gray)
    assert first == second
    assert list(first['scores']) == config.LEGACY_CLASSES
    assert np.isclose(sum(first['scores'].values()), 1.0, atol=1e-4)
    # Regression check against the value measured after the weight-loading fix (CPU inference).
    assert first['predicted_label'] == 'Openness' and abs(first['confidence'] - 0.3678) < 0.002
    # A random-weight or GPU-corrupted model would saturate at ~1.0; the trained model does not.
    assert max(model.predict_gray(np.full((300, 300), 255, np.uint8))['scores'].values()) < 0.95
