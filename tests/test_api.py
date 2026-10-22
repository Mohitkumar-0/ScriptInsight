import io

import numpy as np
import pytest

import config
from app import create_app
from src.services.prediction_service import PredictionService
from tests.conftest import SAMPLE, image_bytes

pytestmark = pytest.mark.skipif(not config.LEGACY_MODEL_PATH.exists(), reason='best_model.h5 not present')


@pytest.fixture(scope='module')
def client():
    return create_app().test_client()  # real service: legacy model loaded, new models not trained


def post(client, data, name, route='/api/predict'):
    return client.post(route, data={'file': (io.BytesIO(data), name)}, content_type='multipart/form-data')


@pytest.mark.parametrize('route', ['/api/predict', '/predict'])
def test_predict_routes_share_schema(client, route):
    r = post(client, SAMPLE.read_bytes(), '1.1.jpg', route)
    assert r.status_code == 200
    body = r.get_json()
    assert set(body) >= {'success', 'prediction', 'model_predictions', 'model_metadata', 'warnings', 'disclaimer'}
    p = body['prediction']
    assert p['model'] == 'legacy_resnet50' and p['task'] == 'multiclass'
    assert set(p['probabilities']) == set(config.TRAITS)
    assert np.isclose(sum(p['probabilities'].values()), 1.0, atol=1e-3)
    statuses = {m['model']: m['status'] for m in body['model_predictions']}
    assert statuses == {'resnet18': 'unavailable', 'svm': 'unavailable', 'random_forest': 'unavailable',
                        'legacy_resnet50': 'ok'}
    assert body['model_metadata']['trait_order'] == config.TRAITS


def test_webp_upload_accepted(client):
    assert post(client, image_bytes('WEBP'), 'sample.webp').status_code == 200


@pytest.mark.parametrize('data,name,status,code', [
    (b'hello', 'notes.txt', 415, 'unsupported_type'),
    (b'not really an image', 'fake.png', 422, 'corrupt_image'),
    (b'', 'empty.png', 400, 'empty_file'),
    (image_bytes('PNG', size=(64, 64)), 'tiny.png', 422, 'too_small'),
])
def test_invalid_uploads(client, data, name, status, code):
    r = post(client, data, name)
    assert r.status_code == status and r.get_json()['code'] == code and r.get_json()['success'] is False


def test_missing_file_field(client):
    r = client.post('/api/predict', data={}, content_type='multipart/form-data')
    assert r.status_code == 400 and r.get_json()['code'] == 'no_file'


def test_oversized_request_returns_json_413(client):
    r = post(client, b'\0' * (config.MAX_UPLOAD_BYTES + 200 * 1024), 'big.jpg')
    assert r.status_code == 413 and r.get_json()['code'] == 'too_large'


def test_uploads_are_not_written_to_disk(client):
    before = sorted(p.name for p in config.UPLOAD_DIR.iterdir())
    post(client, SAMPLE.read_bytes(), '1.1.jpg')
    assert sorted(p.name for p in config.UPLOAD_DIR.iterdir()) == before


def test_info_health_metrics(client):
    info = client.get('/api/info').get_json()
    assert info['traits'] == config.TRAITS and info['primary_model'] == 'legacy_resnet50'
    assert {m['model']: m['state'] for m in info['models']}['svm'] == 'not_trained'
    health = client.get('/api/health')
    assert health.status_code == 200 and health.get_json()['status'] == 'ok'
    metrics = client.get('/api/metrics').get_json()
    assert metrics['legacy_recorded']['test_metrics_recorded']['accuracy'] == 0.303
    assert len(metrics['training_history']['legacy_resnet50']['epochs']) == 49


def test_artifact_route_only_serves_evaluation_outputs(client):
    assert client.get('/artifacts/evaluation/../app.py').status_code == 404
    assert client.get('/artifacts/evaluation/secret.txt').status_code == 404


def test_page_uses_versioned_assets(client):
    html = client.get('/').get_data(as_text=True)
    assert 'js/script.js?v=' in html and 'Experimental' in html


def test_no_models_available_returns_503(tmp_path):
    service = PredictionService(enable_legacy=False, svm_path=tmp_path / 'a', rf_path=tmp_path / 'b',
                                resnet18_path=tmp_path / 'c')
    c = create_app(service).test_client()
    r = post(c, SAMPLE.read_bytes(), '1.1.jpg')
    assert r.status_code == 503 and r.get_json()['code'] == 'model_unavailable'
    assert c.get('/api/health').status_code == 503


class ExplodingService:
    status = {'legacy_resnet50': {'available': True}}
    primary = 'legacy_resnet50'

    def predict(self, gray):
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
