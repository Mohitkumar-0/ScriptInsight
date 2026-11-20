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
# HiEnWrite regression models come last, so they never silently replace a classifier as the default; a client
# can still select one explicitly (``model`` form field of /api/predict).
CLASSIFIER_MODELS = ['resnet18', 'svm', 'random_forest', 'legacy_resnet50']
REGRESSION_MODELS = ['hienwrite_resnet18', 'hienwrite_svr', 'hienwrite_rf']
MODEL_ORDER = CLASSIFIER_MODELS + REGRESSION_MODELS
PRIMARY_ORDER = CLASSIFIER_MODELS + REGRESSION_MODELS
DISPLAY_NAMES = {
    'resnet18': 'ResNet18 (fine-tuned)',
    'svm': 'SVM (HOG + LBP)',
    'random_forest': 'Random Forest (HOG + LBP)',
    'legacy_resnet50': 'Legacy ResNet50 (5-category classifier)',
    'hienwrite_resnet18': 'HiEnWrite ResNet18 regression · experimental',
    'hienwrite_svr': 'HiEnWrite SVR (HOG) · experimental',
    'hienwrite_rf': 'HiEnWrite Random Forest (HOG + LBP) · experimental',
}
REGRESSION_WARNING = ('Experimental HiEnWrite regression: five numerical scores on the dataset\'s questionnaire scale, '
                      'learned from self-reported labels of about 300 writers. Compare its measured test error with '
                      'the train-mean baseline on the Evaluation tab. Not a personality assessment.')
SCORE_TYPE_NOTES = {
    'softmax_probability': 'Softmax over mutually exclusive categories; scores sum to 1. Not calibrated.',
    'sigmoid_probability': 'Independent sigmoid per trait (probability of the high pole). Not calibrated.',
    'platt_probability': 'Platt-scaled SVM probability (cross-validated on the training split).',
    'vote_fraction': 'Fraction of trees voting for the class; not a calibrated probability.',
    'regression_score': 'Predicted score on the HiEnWrite questionnaire scale (training labels range 0-40); '
                        'independent per trait, no confidence attached.',
}


class PredictionService:
    def __init__(self, enable_legacy=config.ENABLE_LEGACY_MODEL, svm_path=config.SVM_PATH, rf_path=config.RF_PATH,
                 resnet18_path=config.RESNET18_PATH, legacy_path=config.LEGACY_MODEL_PATH,
                 hienwrite_resnet18_path=config.HIENWRITE_RESNET18_PATH, hienwrite_svr_path=config.HIENWRITE_SVR_PATH,
                 hienwrite_rf_path=config.HIENWRITE_RF_PATH):
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
        loaders += [
            ('hienwrite_resnet18', lambda: self._resnet18(hienwrite_resnet18_path, regression=True)),
            ('hienwrite_svr', lambda: self._classical(hienwrite_svr_path, regression=True)),
            ('hienwrite_rf', lambda: self._classical(hienwrite_rf_path, regression=True)),
        ]
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
    def _check_task(model, path, regression):
        # The classification slots must never load a regression artifact and vice versa.
        if (model.metadata.get('label_mode') == 'regression') != regression:
            kind = 'HiEnWrite regression' if regression else 'classification'
            raise ModelUnavailableError(f'{Path(path).name} is not a {kind} model.')
        return model

    @classmethod
    def _resnet18(cls, path, regression=False):
        from src.deep_learning.predict_resnet18 import ResNet18Model
        return cls._check_task(ResNet18Model(path), path, regression)

    @classmethod
    def _classical(cls, path, regression=False):
        from src.classical.predict_classical import ClassicalModel
        return cls._check_task(ClassicalModel(path), path, regression)

    @staticmethod
    def _legacy(path):
        from src.deep_learning.legacy_resnet50 import LegacyResNet50Model
        return LegacyResNet50Model(path)

    @property
    def primary(self):
        """Default model. Classifiers first: the best by validation macro-F1 (never test metrics), otherwise the
        first available classifier. Only when no classifier is loaded, the HiEnWrite regressor with the lowest
        validation normalised RMSE."""
        scored = [(m.metadata['validation_metrics']['macro']['f1'], name) for name, m in self.models.items()
                  if 'f1' in ((m.metadata.get('validation_metrics') or {}).get('macro') or {})]
        if scored:
            return max(scored)[1]
        classifier = next((name for name in CLASSIFIER_MODELS if name in self.models), None)
        if classifier:
            return classifier
        regression = [(m.metadata['validation_metrics']['macro']['nrmse'], name) for name, m in self.models.items()
                      if ((m.metadata.get('validation_metrics') or {}).get('macro') or {}).get('nrmse') is not None]
        if regression:
            return min(regression)[1]
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
                    'metadata': {k: v for k, v in meta.items()
                                 if k not in ('validation_metrics', 'label_names', 'search', 'target_mean', 'target_std')},
                    'validation_metrics': meta.get('validation_metrics'),
                    'experimental': bool(meta.get('experimental')),
                    'dataset': meta.get('dataset'),
                })
            info.append(entry)
        return info

    def predict(self, gray, model=None):
        """Run every loaded model. ``model`` selects the primary result (must be loaded); default: ``primary``."""
        if model is not None and model not in self.models:
            raise ModelUnavailableError(f'Model {model} is not available.')
        if not self.models:
            raise ModelUnavailableError('No prediction model is available on the server.')
        results, warnings = [], []
        for name in MODEL_ORDER:
            if name not in self.models:
                results.append({'model': name, 'display_name': DISPLAY_NAMES[name], 'status': 'unavailable',
                                'message': self.status.get(name, {}).get('message', 'Not loaded.')})
                continue
            instance = self.models[name]
            started = time.perf_counter()
            try:
                out = instance.predict_gray(gray)
            except Exception:
                log.exception('Prediction failed for %s', name)
                results.append({'model': name, 'display_name': DISPLAY_NAMES[name], 'status': 'error',
                                'message': 'This model failed on the image. See server log.'})
                warnings.append(f'{DISPLAY_NAMES[name]} failed on this image.')
                continue
            if out.get('task') == 'regression':
                # Reference range of the training labels, so a client can place the scores on the scale.
                out = {**out, 'experimental': True, 'dataset': instance.metadata.get('dataset'),
                       'target_stats': {t: {k: round(v, 2) for k, v in s.items() if k in ('mean', 'min', 'max')}
                                        for t, s in (instance.metadata.get('target_stats') or {}).items()}}
            score_type = instance.metadata.get('score_type')
            results.append({'model': name, 'display_name': DISPLAY_NAMES[name], 'status': 'ok',
                            'score_type': score_type, 'score_note': SCORE_TYPE_NOTES.get(score_type, ''),
                            'latency_ms': round((time.perf_counter() - started) * 1000, 1), **out})

        first = model or self.primary
        preferred = [first] + ([] if model else [n for n in PRIMARY_ORDER if n != first])
        primary = next((r for name in preferred for r in results if r['model'] == name and r['status'] == 'ok'), None)
        if primary is None:
            raise RuntimeError('The selected model failed on this image.' if model else
                               'All available models failed on this image.')
        if primary.get('task') == 'regression':
            warnings.append(REGRESSION_WARNING)
        if primary['model'] == 'legacy_resnet50':
            warnings.append('Only the legacy 5-category model is available. It picks ONE category; the five '
                            'scores sum to 100% and are not independent trait levels.')
        return primary, results, warnings


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        log.exception('Could not read %s', path)
        return None
