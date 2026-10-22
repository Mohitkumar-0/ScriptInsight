"""Image decoding, validation and deterministic preprocessing shared by training and inference.

Inference never applies random augmentation; training-only augmentation lives in
``src.deep_learning.train_resnet18``.
"""
import io
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

import config


class ImageValidationError(ValueError):
    """Raised when an upload is not an acceptable handwriting image."""

    def __init__(self, code, message, http_status=400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status


@dataclass
class DecodedImage:
    gray: np.ndarray          # uint8, H x W, ink dark on light background
    width: int
    height: int
    format: str               # format detected from the file content, e.g. 'JPEG'


def _extension(filename):
    return filename.rsplit('.', 1)[1].lower() if filename and '.' in filename else ''


def decode_image(data, filename='', min_side=None):
    """Validate and decode image bytes. Trusts the decoded content, not the filename or MIME type."""
    min_side = config.MIN_IMAGE_SIDE if min_side is None else min_side

    if not data:
        raise ImageValidationError('empty_file', 'The uploaded file is empty.')
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise ImageValidationError(
            'too_large', f'Image is too large. The limit is {config.MAX_UPLOAD_BYTES // (1024 * 1024)}MB.', 413)
    if filename and _extension(filename) not in config.ALLOWED_EXTENSIONS:
        raise ImageValidationError(
            'unsupported_type', 'Unsupported file type. Allowed: JPG, JPEG, PNG, WEBP, BMP, TIFF, GIF.', 415)

    Image.MAX_IMAGE_PIXELS = config.MAX_IMAGE_PIXELS
    try:
        with Image.open(io.BytesIO(data)) as probe:
            detected = probe.format
            probe.verify()
        image = Image.open(io.BytesIO(data))
        image.load()
    except Image.DecompressionBombError:
        raise ImageValidationError('too_large', 'Image dimensions are too large to process safely.', 413)
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise ImageValidationError(
            'corrupt_image', 'The file could not be decoded as an image. It may be corrupted.', 422)

    if detected not in config.ALLOWED_FORMATS:
        raise ImageValidationError(
            'unsupported_type', f'Unsupported image format ({detected}). Allowed: JPG, PNG, WEBP, BMP, TIFF, GIF.', 415)

    image = ImageOps.exif_transpose(image)  # honour phone-camera orientation
    width, height = image.size
    if width < min_side or height < min_side:
        raise ImageValidationError(
            'too_small',
            f'Image is too small ({width}x{height}). Minimum size is {min_side}x{min_side} pixels.', 422)

    return DecodedImage(gray=_to_gray(image), width=width, height=height, format=detected)


def decode_image_file(path, min_side=None):
    path = Path(path)
    return decode_image(path.read_bytes(), path.name, min_side=min_side)


def _to_gray(image):
    """Convert to 8-bit grayscale, compositing any transparency onto white paper."""
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        rgba = image.convert('RGBA')
        background = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, rgba)
    return np.asarray(image.convert('L'), dtype=np.uint8)


def standardize_gray(gray, size=config.IMAGE_SIZE, denoise=config.DENOISE):
    """Resize to size x size and optionally apply 3x3 Gaussian noise suppression (uint8 output)."""
    if denoise:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
    interpolation = cv2.INTER_AREA if min(gray.shape) > size else cv2.INTER_LINEAR
    return cv2.resize(gray, (size, size), interpolation=interpolation)


def resnet18_tensor(gray_224):
    """Grayscale 224x224 uint8 -> float32 CHW array, 3 identical channels, ImageNet mean/std normalised."""
    x = gray_224.astype(np.float32) / 255.0
    x = np.repeat(x[None, :, :], 3, axis=0)
    mean = np.asarray(config.IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    std = np.asarray(config.IMAGENET_STD, dtype=np.float32)[:, None, None]
    return (x - mean) / std


def legacy_resnet50_batch(gray):
    """Preprocessing used when the legacy Keras ResNet50 was trained: no denoising, linear resize,
    /255 scaling, grayscale stacked into 3 channels. Returns float32 (1, 224, 224, 3)."""
    x = cv2.resize(gray, (config.IMAGE_SIZE, config.IMAGE_SIZE)).astype(np.float32) / 255.0
    return np.stack([x] * 3, axis=-1)[None, ...]
