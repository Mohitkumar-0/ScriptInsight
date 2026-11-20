import io

import numpy as np
import pytest

import config
from app import create_app
from src.services.prediction_service import PredictionService
from tests.conftest import LEGACY_AVAILABLE, SAMPLE, encode, handwriting_like_image, image_bytes, make_regression_artifacts

legacy_only = pytest.mark.skipif(not LEGACY_AVAILABLE, reason='legacy model needs best_model.h5 and a working TensorFlow')
CLASSIFIERS = ['resnet18', 'svm', 'random_forest', 'legacy_resnet50']
REGRESSORS = ['hienwrite_resnet18', 'hienwrite_svr', 'hienwrite_rf']


@pytest.fixture(scope='module')
def client():
    return create_app().test_client()  # real service with whatever artifacts exist on disk


@pytest.fixture(scope='module')
def legacy_client(tmp_path_factory):
    """Real legacy model only: every trained-classifier and HiEnWrite artifact is pointed at a missing file, so the
    result does not depend on which models happen to be trained on this machine."""
    missing = tmp_path_factory.mktemp('none')
    service = PredictionService(svm_path=missing / 'a', rf_path=missing / 'b', resnet18_path=missing / 'c',
                                hienwrite_resnet18_path=missing / 'd', hienwrite_svr_path=missing / 'e',
                                hienwrite_rf_path=missing / 'f')
    return create_app(service).test_client()


def regression_service(tmp_path, names=REGRESSORS):
    """Service with no classifiers and synthetic (untrained) HiEnWrite artifacts for the given names."""
    paths = make_regression_artifacts(tmp_path / 'hienwrite')
    missing = tmp_path / 'missing'
    return PredictionService(
        enable_legacy=False, svm_path=missing / 'a', rf_path=missing / 'b', resnet18_path=missing / 'c',
        **{f'{n}_path': paths[n] if n in names else missing / n for n in REGRESSORS})


@pytest.fixture(scope='module')
def reg_client(tmp_path_factory):
    return create_app(regression_service(tmp_path_factory.mktemp('svc'))).test_client()


def post(client, data, name, route='/api/predict'):
    return client.post(route, data={'file': (io.BytesIO(data), name)}, content_type='multipart/form-data')


@legacy_only
@pytest.mark.parametrize('route', ['/api/predict', '/predict'])
def test_predict_routes_share_schema(legacy_client, route):
    r = post(legacy_client, SAMPLE.read_bytes(), '1.1.jpg', route)
    assert r.status_code == 200
    body = r.get_json()
    assert set(body) >= {'success', 'prediction', 'model_predictions', 'model_metadata', 'warnings', 'disclaimer'}
    p = body['prediction']
    assert p['model'] == 'legacy_resnet50' and p['task'] == 'multiclass'
    assert set(p['probabilities']) == set(config.TRAITS)
    assert np.isclose(sum(p['probabilities'].values()), 1.0, atol=1e-3)
    statuses = {m['model']: m['status'] for m in body['model_predictions']}
    assert {k: statuses[k] for k in CLASSIFIERS} == {'resnet18': 'unavailable', 'svm': 'unavailable',
                                                     'random_forest': 'unavailable', 'legacy_resnet50': 'ok'}
    assert set(statuses) == set(CLASSIFIERS + REGRESSORS)  # HiEnWrite models listed, never the default
    assert body['model_metadata']['trait_order'] == config.TRAITS


def test_webp_upload_accepted(reg_client):
    assert post(reg_client, encode(handwriting_like_image(), 'WEBP'), 'sample.webp').status_code == 200


@pytest.mark.parametrize('data,name,status,code', [
    (b'hello', 'notes.txt', 415, 'unsupported_type'),
    (b'not really an image', 'fake.png', 422, 'corrupt_image'),
    (b'', 'empty.png', 400, 'empty_file'),
    (image_bytes('PNG', size=(64, 64)), 'tiny.png', 422, 'too_small'),
])
def test_invalid_uploads(reg_client, data, name, status, code):
    r = post(reg_client, data, name)
    assert r.status_code == status and r.get_json()['code'] == code and r.get_json()['success'] is False


def test_missing_file_field(reg_client):
    r = reg_client.post('/api/predict', data={}, content_type='multipart/form-data')
    assert r.status_code == 400 and r.get_json()['code'] == 'no_file'


def test_oversized_request_returns_json_413(reg_client):
    r = post(reg_client, b'\0' * (config.MAX_UPLOAD_BYTES + 200 * 1024), 'big.jpg')
    assert r.status_code == 413 and r.get_json()['code'] == 'too_large'


def test_uploads_are_not_written_to_disk(reg_client):
    before = sorted(p.name for p in config.UPLOAD_DIR.iterdir())
    assert post(reg_client, SAMPLE.read_bytes(), '1.1.jpg').status_code == 200
    assert sorted(p.name for p in config.UPLOAD_DIR.iterdir()) == before


@legacy_only
def test_info_health_metrics(legacy_client):
    client = legacy_client
    info = client.get('/api/info').get_json()
    assert info['traits'] == config.TRAITS and info['primary_model'] == 'legacy_resnet50'
    assert {m['model']: m['state'] for m in info['models']}['svm'] == 'not_trained'
    health = client.get('/api/health')
    assert health.status_code == 200 and health.get_json()['status'] == 'ok'
    metrics = client.get('/api/metrics').get_json()
    assert metrics['legacy_recorded']['test_metrics_recorded']['accuracy'] == 0.303
    assert len(metrics['training_history']['legacy_resnet50']['epochs']) == 49


def test_app_with_real_artifacts_serves_predictions(client):
    """Uses whatever models are actually on disk (legacy, HiEnWrite and, once trained, the demo classifiers)."""
    if client.get('/api/health').status_code != 200:
        pytest.skip('no model artifacts on disk')
    body = post(client, SAMPLE.read_bytes(), '1.1.jpg').get_json()
    working = {m['model'] for m in body['model_predictions'] if m['status'] == 'ok'}
    assert body['prediction']['model'] in working


def test_artifact_route_only_serves_evaluation_outputs(reg_client):
    assert reg_client.get('/artifacts/evaluation/../app.py').status_code == 404
    assert reg_client.get('/artifacts/evaluation/secret.txt').status_code == 404
    assert reg_client.get('/artifacts/hienwrite/evaluation/latest.json').status_code == 404  # PNG plots only
    assert reg_client.get('/artifacts/hienwrite/evaluation/../../app.py').status_code == 404


def test_page_uses_versioned_assets(reg_client):
    html = reg_client.get('/').get_data(as_text=True)
    assert 'js/script.js?v=' in html and 'Experimental' not in html


def test_no_models_available_returns_503(tmp_path):
    service = PredictionService(enable_legacy=False, svm_path=tmp_path / 'a', rf_path=tmp_path / 'b',
                                resnet18_path=tmp_path / 'c', hienwrite_resnet18_path=tmp_path / 'd',
                                hienwrite_svr_path=tmp_path / 'e', hienwrite_rf_path=tmp_path / 'f')
    c = create_app(service).test_client()
    r = post(c, SAMPLE.read_bytes(), '1.1.jpg')
    assert r.status_code == 503 and r.get_json()['code'] == 'model_unavailable'
    assert c.get('/api/health').status_code == 503


class ExplodingService:
    status = {'legacy_resnet50': {'available': True}}
    primary = 'legacy_resnet50'

    def predict(self, gray, model=None):
        raise RuntimeError('secret internal detail /Users/someone/path')

    def model_info(self):
        return []


def test_internal_errors_do_not_leak_details():
    r = post(create_app(ExplodingService()).test_client(), SAMPLE.read_bytes(), '1.1.jpg')
    assert r.status_code == 500 and r.get_json()['code'] == 'server_error'
    assert 'secret' not in r.get_data(as_text=True)


class _Stub:
    def __init__(self, f1):
        self.metadata = {'validation_metrics': {'macro': {'f1': f1}}}


def test_primary_model_is_best_by_validation_f1():
    service = PredictionService.__new__(PredictionService)
    service.models = {'svm': _Stub(0.41), 'random_forest': _Stub(0.47), 'resnet18': _Stub(0.44)}
    assert service.primary == 'random_forest'
    service.models = {'legacy_resnet50': type('L', (), {'metadata': {}})()}
    assert service.primary == 'legacy_resnet50'


# ----- HiEnWrite regression models (synthetic, untrained artifacts: API plumbing only) -----
def test_regression_prediction_returns_five_named_scores(reg_client):
    r = post(reg_client, SAMPLE.read_bytes(), '1.1.jpg')
    assert r.status_code == 200
    body = r.get_json()
    p = body['prediction']
    # No classifier is loaded, so the regressor with the best VALIDATION normalised RMSE is the default.
    assert p['model'] == 'hienwrite_svr' and p['task'] == 'regression' and p['experimental'] is True
    assert set(p['trait_scores']) == set(config.TRAITS)  # JSON keys are sorted; clients order by trait_order
    assert all(isinstance(v, float) and np.isfinite(v) for v in p['trait_scores'].values())
    assert p['training_label_stats']['Openness'] == {'mean': 20.0, 'min': 5.0, 'max': 38.0}
    assert any('Experimental HiEnWrite regression' in w for w in body['warnings'])
    assert 'probabilities' not in p and 'confidence' not in p  # never presented as class probabilities


@pytest.mark.parametrize('name', REGRESSORS)
def test_model_selection_picks_requested_model(reg_client, name):
    r = reg_client.post('/api/predict', data={'file': (io.BytesIO(SAMPLE.read_bytes()), '1.1.jpg'), 'model': name},
                        content_type='multipart/form-data')
    assert r.status_code == 200
    body = r.get_json()
    assert body['prediction']['model'] == name and body['model_metadata']['requested_model'] == name
    entry = next(m for m in body['model_predictions'] if m['model'] == name)
    assert entry['status'] == 'ok' and len(entry['scores']) == 5


def test_unknown_and_unavailable_model_requests(reg_client):
    def ask(model):
        return reg_client.post('/api/predict', data={'file': (io.BytesIO(SAMPLE.read_bytes()), '1.1.jpg'), 'model': model},
                               content_type='multipart/form-data')
    r = ask('gpt')
    assert r.status_code == 400 and r.get_json()['code'] == 'unknown_model'
    r = ask('legacy_resnet50')  # known but not loaded: explicit error, never a silent substitute
    assert r.status_code == 503 and r.get_json()['code'] == 'model_unavailable'
    assert ask('auto').status_code == 200


def test_missing_regression_checkpoint_reported_not_faked(tmp_path):
    service = regression_service(tmp_path, names=['hienwrite_svr'])
    assert service.status['hienwrite_resnet18']['state'] == 'not_trained'
    assert 'not found' in service.status['hienwrite_resnet18']['message']
    c = create_app(service).test_client()
    body = post(c, SAMPLE.read_bytes(), '1.1.jpg').get_json()
    statuses = {m['model']: m['status'] for m in body['model_predictions']}
    assert statuses['hienwrite_resnet18'] == 'unavailable' and statuses['hienwrite_svr'] == 'ok'


def test_classifier_slot_rejects_regression_artifact(tmp_path):
    paths = make_regression_artifacts(tmp_path / 'h')
    service = PredictionService(enable_legacy=False, svm_path=paths['hienwrite_svr'], rf_path=tmp_path / 'x',
                                resnet18_path=paths['hienwrite_resnet18'], hienwrite_resnet18_path=tmp_path / 'y',
                                hienwrite_svr_path=tmp_path / 'z', hienwrite_rf_path=tmp_path / 'w')
    assert not service.status['svm']['available'] and not service.status['resnet18']['available']


def test_regression_models_do_not_replace_classifier_default():
    service = PredictionService.__new__(PredictionService)
    regressor = type('R', (), {'metadata': {'validation_metrics': {'macro': {'rmse': 5.0, 'nrmse': 0.5}}}})()
    service.models = {'legacy_resnet50': type('L', (), {'metadata': {}})(), 'hienwrite_svr': regressor}
    assert service.primary == 'legacy_resnet50'


def test_info_and_metrics_include_hienwrite(reg_client):
    info = reg_client.get('/api/info').get_json()
    states = {m['model']: m for m in info['models']}
    assert states['hienwrite_svr']['task'] == 'regression' and states['hienwrite_svr']['experimental'] is True
    assert states['legacy_resnet50']['state'] == 'disabled'
    metrics = reg_client.get('/api/metrics').get_json()
    assert set(metrics['hienwrite']) >= {'evaluation_available', 'evaluation', 'split', 'validation', 'plots'}
    assert reg_client.get('/api/health').get_json()['status'] == 'ok'
