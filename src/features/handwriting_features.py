"""HOG + LBP handcrafted features for the classical (SVM / Random Forest) pipeline.

Parameters are fixed and versioned: a model trained with one FEATURE_VERSION refuses to run on
features produced by another, so training and inference dimensions always agree.

HOG  : 9 orientations, 16x16 px cells, 2x2-cell blocks, L2-Hys -> 13*13*2*2*9 = 6084 values
LBP  : uniform LBP (P=8, R=1) histograms (10 bins) on a 4x4 grid      -> 160 values
       uniform LBP (P=16, R=2) global histogram (18 bins)             ->  18 values
Total: 6262 values per 224x224 grayscale image.
"""
import numpy as np
from skimage.feature import hog, local_binary_pattern

import config
from src.preprocessing.image_processor import standardize_gray

FEATURE_VERSION = 'hog9-c16-b2_lbp8r1-g4_lbp16r2_v1'
HOG_PARAMS = dict(orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm='L2-Hys')
LBP_GRID = 4
HOG_DIM = 6084
LBP_DIM = LBP_GRID * LBP_GRID * 10 + 18
FEATURE_DIM = HOG_DIM + LBP_DIM


def _lbp_histogram(codes, n_bins):
    hist, _ = np.histogram(codes, bins=n_bins, range=(0, n_bins))
    return hist.astype(np.float32) / max(codes.size, 1)


def extract_features(gray_224):
    """Feature vector for an already standardised 224x224 uint8 grayscale image."""
    if gray_224.shape != (config.IMAGE_SIZE, config.IMAGE_SIZE):
        raise ValueError(f'Expected a {config.IMAGE_SIZE}x{config.IMAGE_SIZE} image, got {gray_224.shape}.')

    image = gray_224.astype(np.float32) / 255.0
    hog_vec = hog(image, feature_vector=True, **HOG_PARAMS).astype(np.float32)

    lbp_small = local_binary_pattern(gray_224, P=8, R=1, method='uniform')
    cell = config.IMAGE_SIZE // LBP_GRID
    grid_hists = [
        _lbp_histogram(lbp_small[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell], 10)
        for r in range(LBP_GRID) for c in range(LBP_GRID)
    ]
    lbp_large = local_binary_pattern(gray_224, P=16, R=2, method='uniform')
    features = np.concatenate([hog_vec, *grid_hists, _lbp_histogram(lbp_large, 18)])

    if features.shape != (FEATURE_DIM,) or not np.all(np.isfinite(features)):
        raise ValueError(f'Feature extraction produced an invalid vector of shape {features.shape}.')
    return features


FEATURE_SETS = {'hog': slice(0, HOG_DIM), 'hog_lbp': slice(0, FEATURE_DIM)}


def features_from_gray(gray, denoise=config.DENOISE, fit='stretch'):
    """Raw grayscale image of any size -> feature vector (standardisation + extraction)."""
    return extract_features(standardize_gray(gray, denoise=denoise, fit=fit))


def select_features(features, feature_set):
    """Columns of a full HOG+LBP vector (or matrix) used by a model: 'hog' or 'hog_lbp'."""
    return np.asarray(features)[..., FEATURE_SETS[feature_set]]
