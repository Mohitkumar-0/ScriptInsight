"""Inference for fine-tuned ResNet18 checkpoints (classification or HiEnWrite regression).
Never falls back to random weights."""
from pathlib import Path

import numpy as np
import torch

from src.errors import ModelUnavailableError
from src.deep_learning.model import build_resnet18, outputs_to_scores
from src.preprocessing.image_processor import prepare_gray, resnet18_tensor


class ResNet18Model:
    name = 'resnet18'

    def __init__(self, path):
        path = Path(path)
        if not path.exists():
            raise ModelUnavailableError(f'Model artifact not found: {path.name}. Train it first (see README).')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        meta = checkpoint.get('metadata', {})
        if meta.get('arch') != 'resnet18':
            raise ModelUnavailableError(f'{path.name} is not a ResNet18 checkpoint.')
        model = build_resnet18(meta['num_outputs'], pretrained=False)
        try:
            model.load_state_dict(checkpoint['state_dict'], strict=True)
        except RuntimeError as exc:
            raise ModelUnavailableError(f'{path.name} does not match the ResNet18 architecture: {exc}') from exc
        model.eval()
        self.model = model
        self.metadata = meta
        self.display_name = meta.get('display_name', 'ResNet18')
        self.label_mode = meta['label_mode']
        self.label_names = meta['label_names']
        if self.label_mode == 'regression':
            self.target_mean = np.asarray(meta['target_mean'], dtype=np.float64)
            self.target_std = np.asarray(meta['target_std'], dtype=np.float64)

    def preprocess(self, gray):
        """Model input tensor (1, 3, H, W), prepared exactly as during training."""
        return torch.from_numpy(resnet18_tensor(prepare_gray(gray, self.metadata['preprocessing'])))[None]

    def predict_gray(self, gray):
        with torch.inference_mode():
            outputs = self.model(self.preprocess(gray))
        if self.label_mode == 'regression':
            values = outputs[0].double().numpy() * self.target_std + self.target_mean  # back to the raw score scale
            return {'task': 'regression', 'scores': dict(zip(self.label_names, map(float, values)))}
        scores = dict(zip(self.label_names, map(float, outputs_to_scores(outputs, self.label_mode)[0].tolist())))
        if self.label_mode == 'multilabel':
            return {'task': 'multilabel', 'scores': scores,
                    'levels': {t: 'high' if p >= 0.5 else 'low' for t, p in scores.items()}}
        top = max(scores, key=scores.get)
        return {'task': 'multiclass', 'scores': scores, 'predicted_label': top, 'confidence': scores[top]}
