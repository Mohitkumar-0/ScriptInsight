"""Inference for trained HOG+LBP SVM / Random Forest bundles saved by train_classical."""
from pathlib import Path

import joblib
import numpy as np

from src.errors import ModelUnavailableError
from src.features.handwriting_features import FEATURE_DIM, FEATURE_VERSION, features_from_gray


class ClassicalModel:
    def __init__(self, path):
        path = Path(path)
        if not path.exists():
            raise ModelUnavailableError(f'Model artifact not found: {path.name}. Train it first (see README).')
        # joblib uses pickle: only load artifacts produced locally by train_classical.
        bundle = joblib.load(path)
        if bundle.get('feature_version') != FEATURE_VERSION or bundle.get('feature_dim') != FEATURE_DIM:
            raise ModelUnavailableError(
                f'{path.name} was trained with features {bundle.get("feature_version")} '
                f'({bundle.get("feature_dim")} dims); this code produces {FEATURE_VERSION} ({FEATURE_DIM}). Retrain it.')
        self.bundle = bundle
        self.name = bundle['model_name']
        self.display_name = bundle['display_name']
        self.label_mode = bundle['label_mode']
        self.label_names = bundle['label_names']
        self.metadata = {k: v for k, v in bundle.items() if k != 'estimators'}

    def predict_features(self, features):
        x = np.asarray(features, dtype=np.float32)[None, :]
        if self.label_mode == 'multilabel':
            scores = {}
            for trait, pipe in zip(self.label_names, self.bundle['estimators']):
                scores[trait] = float(pipe.predict_proba(x)[0, list(pipe.classes_).index(1)])
            return {'task': 'multilabel', 'scores': scores,
                    'levels': {t: 'high' if p >= 0.5 else 'low' for t, p in scores.items()}}
        pipe = self.bundle['estimators']
        proba = pipe.predict_proba(x)[0]
        scores = {self.label_names[int(c)]: float(p) for c, p in zip(pipe.classes_, proba)}
        top = max(scores, key=scores.get)
        return {'task': 'multiclass', 'scores': scores, 'predicted_label': top, 'confidence': scores[top]}

    def predict_gray(self, gray):
        return self.predict_features(features_from_gray(gray, denoise=self.bundle['denoise']))
