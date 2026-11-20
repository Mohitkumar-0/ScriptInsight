"""HiEnWrite regression: metrics, preprocessing, classical regressors, ResNet18 regression head, evaluation.

Synthetic data and untrained weights only: these tests check shapes, plumbing and consistency, not accuracy.
The last test loads the REAL trained artifacts when they exist (skipped otherwise)."""
import json

import numpy as np
import pytest
import torch

import config
from src.classical.predict_classical import ClassicalModel
from src.data.hienwrite import TARGETS, read_split_manifest
from src.deep_learning.model import build_resnet18
from src.deep_learning.predict_resnet18 import ResNet18Model
from src.evaluation.metrics import normalized_rmse, regression_metrics
from src.features import handwriting_features as hf
from src.preprocessing import image_processor as ip
from tests.conftest import SAMPLE, make_regression_artifacts, synthetic_writer_image


# ----- metrics -----
def test_regression_metrics_hand_computed():
    y = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, np.nan], [4.0, 40.0]])
    p = np.array([[2.0, 10.0], [2.0, 20.0], [2.0, 99.0], [6.0, 40.0]])
    m = regression_metrics(y, p, ['A', 'B'])
    a = m['per_trait']['A']
    assert a['n'] == 4 and a['mae'] == pytest.approx(1.0) and a['rmse'] == pytest.approx(np.sqrt(6 / 4))
    assert a['r2'] == pytest.approx(1 - 6 / 5)
    assert a['pearson_r'] == pytest.approx(np.corrcoef([1, 2, 3, 4], [2, 2, 2, 6])[0, 1])
    b = m['per_trait']['B']  # the NaN target (masked) is ignored, so B is predicted perfectly
    assert b['n'] == 3 and b['rmse'] == 0.0 and b['r2'] == 1.0 and b['pearson_r'] == pytest.approx(1.0)
    assert m['macro']['rmse'] == pytest.approx((a['rmse'] + 0.0) / 2)
    assert 'unweighted mean' in m['aggregation']


def test_regression_metrics_mask_and_undefined_values():
    y = np.array([[5.0], [6.0], [7.0]])
    constant = regression_metrics(y, np.full((3, 1), 6.0), ['T'])['per_trait']['T']
    assert constant['pearson_r'] is None and constant['pearson_p'] is None  # undefined, not NaN or 0
    masked = regression_metrics(y, y, ['T'], mask=np.array([[True], [False], [False]]))['per_trait']['T']
    assert masked['n'] == 1 and masked['r2'] is None and masked['pearson_r'] is None
    empty = regression_metrics(y, y, ['T'], mask=np.zeros((3, 1), bool))
    assert empty['per_trait']['T']['n'] == 0 and empty['macro']['rmse'] is None


def test_normalized_rmse_against_training_spread():
    m = regression_metrics(np.array([[0.0], [2.0]]), np.array([[1.0], [1.0]]), ['T'])
    assert normalized_rmse(m, {'T': 2.0}) == pytest.approx(0.5)


# ----- preprocessing and features -----
def test_pad_resize_keeps_aspect_ratio():
    wide = np.full((100, 400), 200, np.uint8)
    wide[40:60, :] = 0  # a dark horizontal stroke
    out = ip.standardize_gray(wide, size=224, denoise=False, fit='pad')
    assert out.shape == (224, 224) and out.dtype == np.uint8
    rows_with_ink = np.nonzero((out < 100).any(axis=1))[0]
    assert 10 <= len(rows_with_ink) <= 13            # 20 px stroke scaled by 224/400, not stretched 2.24x
    assert out[0, 0] == 200 and out[-1, -1] == 200   # padding uses the page's median gray
    assert ip.standardize_gray(wide, size=224, denoise=False).shape == (224, 224)  # stretch still default
    with pytest.raises(ValueError):
        ip.standardize_gray(wide, fit='crop')


def test_feature_sets_select_hog_and_full_vector():
    page = synthetic_writer_image(1)[:160]  # landscape, like most HiEnWrite photos
    features = hf.features_from_gray(page, fit='pad')
    assert features.shape == (hf.FEATURE_DIM,)
    assert hf.select_features(features, 'hog').shape == (hf.HOG_DIM,) == (6084,)
    assert hf.select_features(features[None], 'hog_lbp').shape == (1, 6262)
    assert not np.array_equal(features, hf.features_from_gray(page))  # pad != stretch for non-square pages


# ----- classical regressors -----
def test_classical_regression_training_and_inference(hienwrite_split, tmp_path):
    from src.classical.train_classical import save_bundle
    from src.classical.train_regression import QUICK_RF_GRID, QUICK_SVR_GRID, extract_features, train_model

    manifest, _ = hienwrite_split
    samples, meta = read_split_manifest(manifest)
    X = extract_features(samples, cache_dir=tmp_path / 'cache', n_jobs=1)
    assert X.shape == (len(samples), hf.FEATURE_DIM)
    assert np.array_equal(X, extract_features(samples, cache_dir=tmp_path / 'cache', n_jobs=1))  # cache hit
    counts = {s: sum(x['split'] == s for x in samples) for s in ('train', 'val')}
    for kind, grid in (('svr', QUICK_SVR_GRID), ('random_forest', QUICK_RF_GRID)):
        bundle = train_model(kind, samples, meta, X, seed=0, grid=grid)
        assert (bundle['n_train'], bundle['n_val']) == (counts['train'], counts['val'])  # test split never used
        assert len(bundle['estimators']) == 5 and bundle['label_names'] == TARGETS
        hindi_train = sum(s['split'] == 'train' and s['language'] == 'h' for s in samples)
        assert bundle['n_train_per_trait']['Openness'] == counts['train'] - hindi_train
        assert bundle['validation_metrics']['task'] == 'regression'
        path = tmp_path / f'{kind}.joblib'
        save_bundle(bundle, path)
        model = ClassicalModel(path)
        out = model.predict_gray(synthetic_writer_image(5))
        assert out['task'] == 'regression' and list(out['scores']) == TARGETS
        assert all(np.isfinite(v) for v in out['scores'].values())
        assert json.loads(path.with_suffix('.json').read_text())['feature_set'] == bundle['feature_set']


# ----- ResNet18 regression -----
def test_resnet18_five_output_head_loss_and_backward():
    from src.deep_learning.train_resnet18_regression import masked_loss
    model = build_resnet18(5)
    assert isinstance(model, torch.nn.Module) and model.fc.out_features == 5 and model.fc.in_features == 512
    x = torch.randn(3, 3, 224, 224)
    pred = model(x)
    assert pred.shape == (3, 5)
    target = torch.zeros(3, 5)
    mask = torch.ones(3, 5)
    mask[0, 4] = 0  # masked value must not contribute
    loss = masked_loss(pred, target, mask)
    expected = ((pred ** 2) * mask).sum() / mask.sum()
    assert torch.isclose(loss, expected)
    loss.backward()
    assert model.fc.weight.grad is not None and torch.isfinite(model.fc.weight.grad).all()
    assert model.conv1.weight.grad is not None
    changed = target.clone()
    changed[0, 4] = 100.0
    assert torch.isclose(masked_loss(pred.detach(), changed, mask), masked_loss(pred.detach(), target, mask))
    assert masked_loss(pred.detach(), target, mask, kind='smoothl1') > 0


def test_resnet18_regression_training_checkpoint_and_inference(hienwrite_split, tmp_path):
    from src.deep_learning.train_resnet18_regression import RegressionDataset, train

    manifest, _ = hienwrite_split
    checkpoint, history = train(manifest, tmp_path / 'deep', epochs=1, batch_size=8, pretrained=False, seed=0,
                                device='cpu', n_jobs=1, log=lambda *_: None)
    assert len(history) == 1 and np.isfinite(history[0]['train_loss']) and np.isfinite(history[0]['val_loss'])
    model = ResNet18Model(checkpoint)
    meta = model.metadata
    samples, _ = read_split_manifest(manifest)
    assert meta['label_mode'] == 'regression' and meta['num_outputs'] == 5 and meta['label_names'] == TARGETS
    assert meta['n_train'] == sum(s['split'] == 'train' for s in samples) and meta['epochs_run'] == 1
    assert meta['preprocessing']['fit'] == 'pad' and not model.model.training
    assert json.loads((tmp_path / 'deep' / 'resnet18_regression_history.json').read_text())['best_epoch'] == 1

    # Inference preprocessing is exactly the training (non-augmented) preprocessing.
    sample = samples[0]
    ds = RegressionDataset([sample], meta['preprocessing'], np.array(meta['target_mean']), np.array(meta['target_std']),
                           augment=False, n_jobs=1)
    gray = ip.decode_image_file(sample['path'], min_side=1).gray
    assert torch.equal(ds[0][0], model.preprocess(gray)[0])

    out = model.predict_gray(gray)
    assert out['task'] == 'regression' and list(out['scores']) == TARGETS
    with torch.no_grad():
        z = model.model(model.preprocess(gray))[0].double().numpy()
    expected = z * np.array(meta['target_std']) + np.array(meta['target_mean'])  # de-standardised
    assert np.allclose([out['scores'][t] for t in TARGETS], expected)
    assert out == model.predict_gray(gray)


def test_untrained_regression_artifacts_load_and_predict(tmp_path):
    paths = make_regression_artifacts(tmp_path)
    gray = ip.decode_image_file(SAMPLE).gray
    for name, path in paths.items():
        model = ResNet18Model(path) if path.suffix == '.pt' else ClassicalModel(path)
        out = model.predict_gray(gray)
        assert out['task'] == 'regression' and len(out['scores']) == 5, name


def test_evaluation_uses_only_test_split(hienwrite_split, tmp_path):
    from src.evaluation.evaluate_regression import evaluate
    manifest, meta = hienwrite_split
    paths = make_regression_artifacts(tmp_path / 'models')
    (tmp_path / 'models' / 'svr_hog.joblib').write_bytes(paths['hienwrite_svr'].read_bytes())
    report = evaluate(manifest, tmp_path / 'eval', tmp_path / 'models', n_bootstrap=50)
    assert report['n_test'] == meta['counts']['test']['samples']
    assert set(report['models']) == {'hienwrite_resnet18', 'hienwrite_svr'} and 'hienwrite_rf' in report['skipped']
    assert any('fingerprint' in w for w in report['warnings'])  # synthetic artifacts were not trained on this split
    base = report['baseline']['test_metrics']['per_trait']['Extraversion']
    assert base['pearson_r'] is None and base['n'] == report['n_test']
    ci = report['bootstrap']['models']['hienwrite_svr']['macro']['rmse_minus_baseline_95ci']
    assert ci[0] <= ci[1]
    preds = (tmp_path / 'eval' / 'predictions_hienwrite_svr.csv').read_text().splitlines()
    assert len(preds) == report['n_test'] + 1 and 'pred_Openness' in preds[0]
    assert (tmp_path / 'eval' / 'latest.json').exists() and 'test_rmse_by_trait.png' in report['plots']


def _trained_artifacts_usable():
    import sklearn
    paths = (config.HIENWRITE_RESNET18_PATH, config.HIENWRITE_SVR_PATH, config.HIENWRITE_RF_PATH)
    if not all(p.exists() for p in paths):
        return False
    # Pickled scikit-learn bundles only load with the versions they were trained with.
    return json.loads(config.HIENWRITE_SVR_PATH.with_suffix('.json').read_text())['sklearn_version'] == sklearn.__version__


@pytest.mark.skipif(not _trained_artifacts_usable(),
                    reason='trained HiEnWrite models not present, or trained with another scikit-learn version')
def test_trained_hienwrite_artifacts_load_and_predict():
    gray = ip.decode_image_file(SAMPLE).gray
    for path, cls in ((config.HIENWRITE_RESNET18_PATH, ResNet18Model), (config.HIENWRITE_SVR_PATH, ClassicalModel),
                      (config.HIENWRITE_RF_PATH, ClassicalModel)):
        model = cls(path)
        assert model.metadata['dataset'] == 'HiEnWrite' and model.metadata['experimental'] is True
        assert not model.metadata.get('smoke_test')
        out = model.predict_gray(gray)
        assert list(out['scores']) == TARGETS and all(0 <= v <= 60 for v in out['scores'].values())
