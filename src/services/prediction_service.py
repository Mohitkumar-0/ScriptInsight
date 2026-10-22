"""Loads every available model once and runs predictions for the API.

A model that is missing or incompatible is reported as unavailable with the reason; it is never
replaced by random weights, copied outputs or placeholder scores.
"""
import json
import logging
import time
from pathlib import Path

import config
from src.errors import ModelUnavailableError

log = logging.getLogger(__name__)

# Display order in the comparison panel. The first available entry of PRIMARY_ORDER is the primary model.
MODEL_ORDER = ['resnet18', 'svm', 'random_forest', 'legacy_resnet50']
PRIMARY_ORDER = ['resnet18', 'svm', 'random_forest', 'legacy_resnet50']
DISPLAY_NAMES = {
    'resnet18': 'ResNet18 (fine-tuned)',
    'svm': 'SVM (HOG + LBP)',
    'random_forest': 'Random Forest (HOG + LBP)',
    'legacy_resnet50': 'Legacy ResNet50 (5-category classifier)',
}
SCORE_TYPE_NOTES = {
    'softmax_probability': 'Softmax over mutually exclusive categories; scores sum to 1. Not calibrated.',
    'sigmoid_probability': 'Independent sigmoid per trait (probability of the high pole). Not calibrated.',
    'platt_probability': 'Platt-scaled SVM probability (cross-validated on the training split).',
    'vote_fraction': 'Fraction of trees voting for the class; not a calibrated probability.',
}


class PredictionService:
    def __init__(self, enable_legacy=config.ENABLE_LEGACY_MODEL, svm_path=config.SVM_PATH, rf_path=config.RF_PATH,
                 resnet18_path=config.RESNET18_PATH, legacy_path=config.LEGACY_MODEL_PATH):
        self.models = {}
        self.status = {}
        loaders = [
            ('resnet18', lambda: self._resnet18(resnet18_path)),
            ('svm', lambda: self._classical(svm_path)),
            ('random_forest', lambda: self._classical(rf_path)),
        ]
        if enable_legacy:
            loaders.append(('legacy_resnet50', lambda: self._legacy(legacy_path)))
        else:
            self.status['legacy_resnet50'] = {'available': False, 'state': 'disabled',
                                              'message': 'Disabled by SCRIPTINSIGHT_ENABLE_LEGACY_MODEL.'}
        for name, load in loaders:
            started = time.perf_counter()
            try:
                self.models[name] = load()
                self.status[name] = {'available': True, 'state': 'loaded',
                                     'load_seconds': round(time.perf_counter() - started, 2)}
                log.info('Loaded model %s', name)
            except ModelUnavailableError as exc:
                self.status[name] = {'available': False, 'state': 'not_trained', 'message': str(exc)}
                log.warning('Model %s unavailable: %s', name, exc)
            except Exception as exc:  # incompatible artifact or framework problem
                self.status[name] = {'available': False, 'state': 'error',
                                     'message': f'Failed to load ({exc.__class__.__name__}). See server log.'}
                log.exception('Model %s failed to load', name)

    @staticmethod
    def _resnet18(path):
        from src.deep_learning.predict_resnet18 import ResNet18Model
        return ResNet18Model(path)

    @staticmethod
    def _classical(path):
        from src.classical.predict_classical import ClassicalModel
        return ClassicalModel(path)

    @staticmethod
    def _legacy(path):
        from src.deep_learning.legacy_resnet50 import LegacyResNet50Model
        return LegacyResNet50Model(path)

    @property
    def primary(self):
        """Best trained model by validation macro-F1 (never test metrics); otherwise the first of PRIMARY_ORDER."""
        scored = [(m.metadata['validation_metrics']['macro']['f1'], name) for name, m in self.models.items()
                  if (m.metadata.get('validation_metrics') or {}).get('macro')]
        if scored:
            return max(scored)[1]
        return next((name for name in PRIMARY_ORDER if name in self.models), None)

    def model_info(self):
        info = []
        for name in MODEL_ORDER:
            entry = {'model': name, 'display_name': DISPLAY_NAMES[name], **self.status.get(name, {})}
            if name in self.models:
                meta = self.models[name].metadata
                entry.update({
                    'task': meta.get('label_mode'), 'labels': meta.get('label_names'),
                    'score_type': meta.get('score_type'),
                    'score_note': SCORE_TYPE_NOTES.get(meta.get('score_type'), ''),
                    'metadata': {k: v for k, v in meta.items() if k not in ('validation_metrics', 'label_names')},
                    'validation_metrics': meta.get('validation_metrics'),
                })
            info.append(entry)
        return info

    def predict(self, gray):
        if not self.models:
            raise ModelUnavailableError('No prediction model is available on the server.')
        results, warnings = [], []
        for name in MODEL_ORDER:
            if name not in self.models:
                results.append({'model': name, 'display_name': DISPLAY_NAMES[name], 'status': 'unavailable',
                                'message': self.status.get(name, {}).get('message', 'Not loaded.')})
                continue
            model = self.models[name]
            started = time.perf_counter()
            try:
                out = model.predict_gray(gray)
            except Exception:
                log.exception('Prediction failed for %s', name)
                results.append({'model': name, 'display_name': DISPLAY_NAMES[name], 'status': 'error',
                                'message': 'This model failed on the image. See server log.'})
                warnings.append(f'{DISPLAY_NAMES[name]} failed on this image.')
                continue
            score_type = model.metadata.get('score_type')
            results.append({'model': name, 'display_name': DISPLAY_NAMES[name], 'status': 'ok',
                            'score_type': score_type, 'score_note': SCORE_TYPE_NOTES.get(score_type, ''),
                            'latency_ms': round((time.perf_counter() - started) * 1000, 1), **out})

        preferred = [self.primary] + [n for n in PRIMARY_ORDER if n != self.primary]
        primary = next((r for name in preferred for r in results if r['model'] == name and r['status'] == 'ok'), None)
        if primary is None:
            raise RuntimeError('All available models failed on this image.')
        if primary['model'] == 'legacy_resnet50':
            warnings.append('Only the legacy 5-category model is available. It picks ONE category; the five '
                            'scores sum to 100% and are not independent trait levels.')
        return primary, results, warnings


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        log.exception('Could not read %s', path)
        return None
