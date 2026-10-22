"""Inference for the fine-tuned ResNet18 checkpoint. Never falls back to random weights."""
from pathlib import Path

import torch

from src.errors import ModelUnavailableError
from src.deep_learning.model import build_resnet18, outputs_to_scores
from src.preprocessing.image_processor import resnet18_tensor, standardize_gray


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

    def predict_gray(self, gray):
        prep = self.metadata['preprocessing']
        x = torch.from_numpy(resnet18_tensor(standardize_gray(gray, size=prep['size'], denoise=prep['denoise'])))
        with torch.inference_mode():
            scores = outputs_to_scores(self.model(x[None]), self.label_mode)[0].tolist()
        scores = dict(zip(self.label_names, map(float, scores)))
        if self.label_mode == 'multilabel':
            return {'task': 'multilabel', 'scores': scores,
                    'levels': {t: 'high' if p >= 0.5 else 'low' for t, p in scores.items()}}
        top = max(scores, key=scores.get)
        return {'task': 'multiclass', 'scores': scores, 'predicted_label': top, 'confidence': scores[top]}
