"""ResNet18 (torchvision) with its 1000-class ImageNet head replaced by a 512 -> N linear layer.

multilabel : N = 5 trait logits, trained with BCEWithLogitsLoss, read with a sigmoid per trait.
multiclass : N = number of categories, trained with CrossEntropyLoss, read with a softmax.
"""
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

PRETRAINED_TAG = 'torchvision ResNet18_Weights.IMAGENET1K_V1'


def build_resnet18(num_outputs, pretrained=False, weights_path=None, freeze_backbone=False):
    """Build ResNet18. ``pretrained`` downloads ImageNet weights via torchvision (cached in ~/.cache/torch);
    ``weights_path`` loads the same ImageNet state dict from a local file for offline use."""
    if weights_path:
        model = resnet18(weights=None)
        model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
    elif pretrained:
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = resnet18(weights=None)

    model.fc = nn.Linear(model.fc.in_features, num_outputs)  # 512 -> num_outputs
    if freeze_backbone:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith('fc.')
    return model


def outputs_to_scores(logits, label_mode):
    """Logits -> per-label scores in [0, 1] according to the head's loss."""
    return torch.sigmoid(logits) if label_mode == 'multilabel' else torch.softmax(logits, dim=-1)
