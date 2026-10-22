import numpy as np
import pytest

import config
from src.preprocessing import image_processor as ip
from tests.conftest import SAMPLE, image_bytes


@pytest.mark.parametrize('fmt,ext', [('JPEG', 'jpg'), ('JPEG', 'jpeg'), ('PNG', 'png'), ('WEBP', 'webp'),
                                     ('BMP', 'bmp'), ('TIFF', 'tiff'), ('GIF', 'gif')])
def test_supported_formats_decode(fmt, ext):
    decoded = ip.decode_image(image_bytes(fmt), f'sample.{ext}')
    assert decoded.format == fmt
    assert decoded.gray.dtype == np.uint8 and decoded.gray.shape == (300, 300)


def test_real_sample_decodes():
    decoded = ip.decode_image_file(SAMPLE)
    assert (decoded.width, decoded.height) == (900, 1600)


def test_unsupported_extension_rejected():
    with pytest.raises(ip.ImageValidationError) as exc:
        ip.decode_image(b'hello', 'notes.txt')
    assert exc.value.code == 'unsupported_type' and exc.value.http_status == 415


def test_content_is_validated_not_filename():
    with pytest.raises(ip.ImageValidationError) as exc:
        ip.decode_image(b'this is not an image', 'fake.png')
    assert exc.value.code == 'corrupt_image' and exc.value.http_status == 422
    # A real JPEG with a .png name is accepted, and the detected content format is reported.
    assert ip.decode_image(image_bytes('JPEG'), 'misnamed.png').format == 'JPEG'


def test_truncated_image_rejected():
    data = image_bytes('PNG')
    with pytest.raises(ip.ImageValidationError) as exc:
        ip.decode_image(data[: len(data) // 2], 'cut.png')
    assert exc.value.code == 'corrupt_image'


def test_empty_file_rejected():
    with pytest.raises(ip.ImageValidationError) as exc:
        ip.decode_image(b'', 'empty.png')
    assert exc.value.code == 'empty_file'


def test_minimum_dimensions():
    with pytest.raises(ip.ImageValidationError) as exc:
        ip.decode_image(image_bytes('PNG', size=(64, 400)), 'thin.png')
    assert exc.value.code == 'too_small' and exc.value.http_status == 422
    ip.decode_image(image_bytes('PNG', size=(config.MIN_IMAGE_SIDE, config.MIN_IMAGE_SIDE)), 'edge.png')


def test_size_limit(monkeypatch):
    monkeypatch.setattr(config, 'MAX_UPLOAD_BYTES', 1000)
    with pytest.raises(ip.ImageValidationError) as exc:
        ip.decode_image(image_bytes('BMP'), 'big.bmp')
    assert exc.value.code == 'too_large' and exc.value.http_status == 413


def test_transparent_png_composited_on_white():
    gray = ip.decode_image(image_bytes('PNG', mode='RGBA'), 't.png').gray
    assert gray[150, 5] == 255  # transparent background becomes white paper, not black


def test_standardize_shape_and_determinism():
    gray = ip.decode_image_file(SAMPLE).gray
    a = ip.standardize_gray(gray)
    assert a.shape == (224, 224) and a.dtype == np.uint8
    assert np.array_equal(a, ip.standardize_gray(gray))
    assert not np.array_equal(a, ip.standardize_gray(gray, denoise=False))


def test_resnet18_tensor_normalisation():
    x = ip.resnet18_tensor(np.full((224, 224), 255, np.uint8))
    assert x.shape == (3, 224, 224) and x.dtype == np.float32
    expected = (1 - np.array(config.IMAGENET_MEAN)) / np.array(config.IMAGENET_STD)
    assert np.allclose(x[:, 0, 0], expected, atol=1e-6)


def test_legacy_batch_matches_training_preprocessing():
    x = ip.legacy_resnet50_batch(ip.decode_image_file(SAMPLE).gray)
    assert x.shape == (1, 224, 224, 3) and x.dtype == np.float32
    assert 0.0 <= x.min() and x.max() <= 1.0
    assert np.array_equal(x[..., 0], x[..., 2])  # grayscale stacked into 3 identical channels
