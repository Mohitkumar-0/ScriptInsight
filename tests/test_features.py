import numpy as np
import pytest

from src.features import handwriting_features as hf
from src.preprocessing.image_processor import decode_image_file, standardize_gray
from tests.conftest import SAMPLE, synthetic_writer_image


def test_feature_vector_dimension_and_finiteness():
    features = hf.features_from_gray(decode_image_file(SAMPLE).gray)
    assert features.shape == (hf.FEATURE_DIM,) == (6262,)
    assert features.dtype == np.float32 and np.all(np.isfinite(features))


def test_feature_parts_have_documented_sizes():
    assert hf.HOG_DIM == 13 * 13 * 2 * 2 * 9
    assert hf.LBP_DIM == 4 * 4 * 10 + 18
    lbp = hf.extract_features(standardize_gray(synthetic_writer_image(3)))[hf.HOG_DIM:]
    # each LBP histogram is normalised, so the 16 grid cells + 1 global histogram each sum to 1
    assert np.isclose(lbp.sum(), 17.0, atol=1e-4)


def test_features_are_deterministic_and_image_dependent():
    a = hf.features_from_gray(synthetic_writer_image(1))
    assert np.array_equal(a, hf.features_from_gray(synthetic_writer_image(1)))
    assert not np.array_equal(a, hf.features_from_gray(synthetic_writer_image(2)))


def test_wrong_input_size_rejected():
    with pytest.raises(ValueError):
        hf.extract_features(np.zeros((100, 100), np.uint8))
