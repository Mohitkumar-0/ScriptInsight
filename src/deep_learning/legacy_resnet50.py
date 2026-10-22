"""Legacy Keras ResNet50 demo model (``best_model.h5``), kept until a verified replacement is trained.

The H5 file was saved by Keras 3.12, which TensorFlow 2.13 cannot deserialize, so the notebook's
architecture is rebuilt here and the trained weights are loaded by layer name. It is a single-label,
5-category softmax classifier: the five scores sum to 1 and are NOT independent trait levels.
"""
from pathlib import Path

import numpy as np

import config
from src.errors import ModelUnavailableError
from src.preprocessing.image_processor import legacy_resnet50_batch


def _build(keras):
    base_model = keras.applications.ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
    return keras.Sequential([
        keras.layers.Input(shape=(224, 224, 3)),
        base_model,
        keras.layers.GlobalAveragePooling2D(name='global_average_pooling2d'),
        keras.layers.Dense(512, activation='relu', name='dense_1'),
        keras.layers.BatchNormalization(name='batch_normalization'),
        keras.layers.Dropout(0.5, name='dropout'),
        keras.layers.Dense(256, activation='relu', name='dense_2'),
        keras.layers.BatchNormalization(name='batch_normalization_1'),
        keras.layers.Dropout(0.4, name='dropout_1'),
        keras.layers.Dense(128, activation='relu', name='dense_3'),
        keras.layers.BatchNormalization(name='batch_normalization_2'),
        keras.layers.Dropout(0.3, name='dropout_2'),
        keras.layers.Dense(len(config.LEGACY_CLASSES), activation='softmax', name='output')
    ])


def _load_keras3_h5_weights(target_model, path):
    import h5py

    def collect(group):
        weights = {}
        group.visititems(lambda name, obj: weights.__setitem__(name.split('/')[-1], obj[()])
                         if isinstance(obj, h5py.Dataset) else None)
        return weights

    with h5py.File(path, 'r') as f:
        saved = f['model_weights']
        base_model = target_model.get_layer('resnet50')
        layers = [(layer, saved[layer.name]) for layer in target_model.layers
                  if layer.weights and layer.name != 'resnet50']
        layers += [(layer, saved['resnet50'][layer.name]) for layer in base_model.layers if layer.weights]
        for layer, group in layers:
            stored = collect(group)
            layer.set_weights([stored[w.name.split('/')[-1].split(':')[0]] for w in layer.weights])


class LegacyResNet50Model:
    name = 'legacy_resnet50'
    display_name = 'Legacy ResNet50 (5-category classifier)'
    label_mode = 'multiclass'
    label_names = list(config.LEGACY_CLASSES)

    def __init__(self, path=config.LEGACY_MODEL_PATH):
        path = Path(path)
        if not path.exists():
            raise ModelUnavailableError(f'Legacy model file not found: {path.name}')
        import tensorflow as tf
        from tensorflow import keras
        # tensorflow-metal returns corrupted (saturated) outputs from model.predict for this model on the
        # Apple GPU; CPU inference is correct and fast enough for single images.
        tf.config.set_visible_devices([], 'GPU')
        try:
            self.model = keras.models.load_model(path, compile=False)
        except Exception:
            self.model = _build(keras)
            try:
                _load_keras3_h5_weights(self.model, path)
            except Exception as exc:
                raise ModelUnavailableError(f'Could not load trained weights from {path.name}: {exc}') from exc
        self.metadata = {
            'model_name': self.name, 'display_name': self.display_name, 'arch': 'Keras ResNet50 + dense head',
            'label_mode': self.label_mode, 'label_names': self.label_names, 'score_type': 'softmax_probability',
            'artifact': path.name, 'parameters': int(self.model.count_params()),
            'backbone': 'ImageNet ResNet50, frozen during the saved checkpoint',
            'preprocessing': 'grayscale, 224x224 linear resize, /255, stacked to 3 channels (no denoise)',
            'provenance': 'Trained in personality_prediction_resnet.ipynb; dataset not available for re-evaluation.',
        }

    def predict_gray(self, gray):
        probabilities = self.model.predict(legacy_resnet50_batch(gray), verbose=0)[0]
        scores = {name: float(p) for name, p in zip(self.label_names, probabilities)}
        top = self.label_names[int(np.argmax(probabilities))]
        return {'task': 'multiclass', 'scores': scores, 'predicted_label': top, 'confidence': scores[top]}
