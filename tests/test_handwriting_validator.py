"""Handwriting-content validation (src/preprocessing/handwriting_validator.py) and its use in the API.

The validator is heuristic. Tests marked "heuristic" check its behaviour on particular images; they do not
show that it recognises image content in general (see docs/HANDWRITING_VALIDATION.md for its limits).
"""
import io

import cv2
import numpy as np
import pytest
from PIL import Image

from app import create_app
from src.preprocessing.handwriting_validator import validate_handwriting
from src.preprocessing.image_processor import decode_image
from tests.conftest import SAMPLE, encode, handwriting_like_image
from tests.test_api import post, regression_service

skimage_data = pytest.importorskip('skimage.data')


def check(rgb):
    return validate_handwriting(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), rgb)


def rgb_of(array):
    return np.stack([array] * 3, axis=-1) if array.ndim == 2 else array[..., :3]


# ----- Validator -----
def test_real_handwriting_sample_accepted():
    decoded = decode_image(SAMPLE.read_bytes(), SAMPLE.name)
    result = validate_handwriting(decoded.gray, decoded.rgb)
    assert result.valid and result.code == 'ok'


@pytest.mark.parametrize('variant', ['plain', 'unruled', 'light_on_dark', 'grayscale_only', 'large_page'])
def test_synthetic_handwriting_variants_accepted(variant):
    """Heuristic: cursive-font pages in several forms keep the stroke/line structure the validator needs."""
    page = handwriting_like_image(size=(1800, 1200) if variant == 'large_page' else (900, 600),
                                  ruled=variant != 'unruled')
    if variant == 'light_on_dark':
        page = 255 - page
    if variant == 'grayscale_only':
        assert validate_handwriting(cv2.cvtColor(page, cv2.COLOR_RGB2GRAY)).valid
    else:
        assert check(page).valid


@pytest.mark.parametrize('value', [255, 0, 128])
def test_blank_images_rejected(value):
    result = check(np.full((400, 400, 3), value, np.uint8))
    assert not result.valid and result.code == 'blank_image'


def test_nearly_uniform_paper_with_sensor_noise_rejected():
    rng = np.random.default_rng(0)
    paper = np.clip(250 + rng.normal(0, 3, (400, 400, 3)), 0, 255).astype(np.uint8)
    assert check(paper).code == 'blank_image'


@pytest.mark.parametrize('crop', [None, (slice(20, 260), slice(120, 330))], ids=['portrait', 'face_close_up'])
def test_face_photo_rejected(crop):
    """Heuristic: OpenCV's frontal-face + eye cascades find this (bundled) portrait photo. Faces that are
    turned, small, covered or poorly lit may not be detected; those images rely on the other checks."""
    photo = skimage_data.astronaut()
    if crop:
        photo = np.ascontiguousarray(photo[crop])
    result = check(photo)
    assert not result.valid and result.code == 'face_detected'


@pytest.mark.parametrize('name', ['coffee', 'chelsea', 'rocket', 'camera', 'grass', 'gravel', 'hubble_deep_field'])
def test_photographs_and_textures_rejected(name):
    """Heuristic: these bundled photos lack pen-stroke/text-line structure. Not every photo is caught (e.g. the
    'coins' and 'moon' samples are accepted); see the documented limitations."""
    result = check(rgb_of(getattr(skimage_data, name)()))
    assert not result.valid and result.code == 'no_handwriting'


def test_scenery_photo_rejected():
    """Heuristic: scikit-learn's bundled scenery photo (china.jpg)."""
    from sklearn.datasets import load_sample_image
    assert check(load_sample_image('china.jpg')).code == 'no_handwriting'


@pytest.mark.parametrize('pattern', ['noise', 'gradient', 'checkerboard'])
def test_synthetic_non_writing_rejected(pattern):
    rng = np.random.default_rng(0)
    if pattern == 'noise':
        gray = rng.integers(0, 256, (400, 400)).astype(np.uint8)
    elif pattern == 'gradient':
        gray = np.tile(np.linspace(0, 255, 400), (400, 1)).astype(np.uint8)
    else:
        gray = skimage_data.checkerboard()
    assert not check(rgb_of(gray)).valid


# ----- API -----
@pytest.fixture
def spied_client(tmp_path, monkeypatch):
    """Client whose prediction service records every call to predict()."""
    service = regression_service(tmp_path)
    calls = []
    original = service.predict

    def spy(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(service, 'predict', spy)
    return create_app(service).test_client(), calls


def _photo_bytes():
    return encode(skimage_data.coffee(), 'JPEG')


@pytest.mark.parametrize('route', ['/api/predict', '/predict'])
@pytest.mark.parametrize('kind,reason', [('white', 'blank_image'), ('black', 'blank_image'),
                                         ('face', 'face_detected'), ('photo', 'no_handwriting')])
def test_invalid_images_return_400_and_skip_models(spied_client, route, kind, reason):
    client, calls = spied_client
    data = {'white': encode(np.full((300, 300, 3), 255, np.uint8)),
            'black': encode(np.zeros((300, 300, 3), np.uint8)),
            'face': encode(skimage_data.astronaut(), 'JPEG'),
            'photo': _photo_bytes()}[kind]
    r = post(client, data, f'{kind}.png', route)
    assert r.status_code == 400
    body = r.get_json()
    assert body['success'] is False and body['code'] == 'invalid_handwriting_image' and body['reason'] == reason
    assert body['message'] == body['error'] and 'No handwriting was detected' in body['message']
    assert body['detail']
    assert not {'prediction', 'model_predictions', 'trait_scores'} & set(body)  # no scores of any kind
    assert calls == []


@pytest.mark.parametrize('model', ['hienwrite_resnet18', 'hienwrite_svr', 'hienwrite_rf'])
def test_explicit_model_selection_is_also_validated(spied_client, model):
    client, calls = spied_client
    r = client.post('/api/predict', data={'file': (io.BytesIO(_photo_bytes()), 'photo.jpg'), 'model': model},
                    content_type='multipart/form-data')
    assert r.status_code == 400 and r.get_json()['code'] == 'invalid_handwriting_image'
    assert calls == []


def test_undecodable_image_rejected_before_models(spied_client):
    client, calls = spied_client
    r = post(client, b'\x89PNG\r\n\x1a\n not really a png', 'broken.png')
    assert r.status_code == 422 and r.get_json()['code'] == 'corrupt_image'
    assert calls == []


@pytest.mark.parametrize('source', ['sample', 'synthetic'])
def test_valid_handwriting_reaches_models_with_unchanged_response(spied_client, source):
    client, calls = spied_client
    data = SAMPLE.read_bytes() if source == 'sample' else encode(handwriting_like_image(), 'PNG')
    r = post(client, data, f'{source}.png')
    assert r.status_code == 200 and len(calls) == 1
    body = r.get_json()
    assert set(body) == {'success', 'prediction', 'model_predictions', 'image', 'model_metadata', 'warnings',
                         'disclaimer'}
    assert body['success'] is True and body['prediction']['task'] == 'regression'


def test_transparent_png_handwriting_validated_on_white(spied_client):
    """Alpha is flattened onto white for the colour image too, so transparent scans are not treated as dark."""
    client, calls = spied_client
    page = handwriting_like_image()
    alpha = np.where(page.mean(axis=2) < 150, 255, 0).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(np.dstack([page, alpha])).save(buffer, format='PNG')
    assert post(client, buffer.getvalue(), 'transparent.png').status_code == 200 and len(calls) == 1
