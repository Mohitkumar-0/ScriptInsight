"""Fine-tune ResNet18 on a split manifest.

* Augmentation (train split only): RandomAffine(rotation <= 10 deg, translate 3%, scale 0.95-1.05,
  shear 5 deg, white fill). No horizontal flip: mirrored handwriting is not a realistic sample.
* Validation and test images get only the deterministic preprocessing used at inference.
* The best checkpoint is chosen by validation macro-F1 (fixed in advance); the test split is not read.
* Loss: BCEWithLogitsLoss with positive-class weights (multilabel) or class-weighted CrossEntropyLoss
  (multiclass), computed from training-split label frequencies.

Usage:
    python -m src.deep_learning.train_resnet18 --manifest artifacts/splits/manifest.csv --epochs 25
"""
import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import config
from src.data.dataset import read_manifest
from src.deep_learning.model import PRETRAINED_TAG, build_resnet18, outputs_to_scores
from src.evaluation.metrics import multiclass_metrics, multilabel_metrics, selection_score
from src.preprocessing.image_processor import decode_image_file, resnet18_tensor, standardize_gray

TRAIN_AUGMENTATION = transforms.RandomAffine(degrees=10, translate=(0.03, 0.03), scale=(0.95, 1.05),
                                             shear=5, fill=255)


class HandwritingDataset(Dataset):
    def __init__(self, samples, label_mode, denoise, augment):
        self.label_mode = label_mode
        self.augment = augment
        # Deterministic preprocessing is done once; only the random augmentation runs per epoch.
        self.images = [standardize_gray(decode_image_file(s['path'], min_side=1).gray, denoise=denoise) for s in samples]
        self.labels = [s['labels'] for s in samples]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.augment:
            image = np.asarray(TRAIN_AUGMENTATION(Image.fromarray(image)), dtype=np.uint8)
        x = torch.from_numpy(resnet18_tensor(image))
        y = (torch.tensor(self.labels[idx], dtype=torch.float32) if self.label_mode == 'multilabel'
             else torch.tensor(self.labels[idx], dtype=torch.long))
        return x, y


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name):
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def make_loss(label_mode, train_labels, num_outputs, device):
    if label_mode == 'multilabel':
        y = np.asarray(train_labels, dtype=np.float32)
        positives = y.sum(0)
        pos_weight = (len(y) - positives) / np.maximum(positives, 1)
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    counts = np.bincount(train_labels, minlength=num_outputs).astype(np.float32)
    weights = len(train_labels) / (num_outputs * np.maximum(counts, 1))
    return nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    model.train(optimizer is not None)
    total, count, outputs, targets = 0.0, 0, [], []
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += float(loss) * len(x)
            count += len(x)
            outputs.append(logits.detach().cpu())
            targets.append(y.cpu())
    return total / max(count, 1), torch.cat(outputs), torch.cat(targets)


def compute_metrics(logits, targets, label_mode, label_names):
    scores = outputs_to_scores(logits, label_mode).numpy()
    if label_mode == 'multilabel':
        return multilabel_metrics(targets.numpy().astype(int), scores, label_names)
    return multiclass_metrics(targets.numpy(), scores.argmax(1), label_names)


def train(manifest, out_dir=config.DEEP_DIR, epochs=25, batch_size=32, lr=1e-4, pretrained=True,
          weights_path=None, freeze_backbone=False, denoise=config.DENOISE, seed=config.SEED, device='auto'):
    seed_everything(seed)
    samples, meta = read_manifest(manifest)
    label_mode, label_names = meta['label_mode'], meta['label_names']
    num_outputs = len(label_names)
    device = pick_device(device)

    train_samples = [s for s in samples if s['split'] == 'train']
    val_samples = [s for s in samples if s['split'] == 'val']
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(HandwritingDataset(train_samples, label_mode, denoise, augment=True),
                              batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(HandwritingDataset(val_samples, label_mode, denoise, augment=False), batch_size=batch_size)

    model = build_resnet18(num_outputs, pretrained=pretrained, weights_path=weights_path,
                           freeze_backbone=freeze_backbone).to(device)
    loss_fn = make_loss(label_mode, [s['labels'] for s in train_samples], num_outputs, device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history, best = [], None
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_loss, _, _ = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss, val_logits, val_targets = run_epoch(model, val_loader, loss_fn, device)
        val_metrics = compute_metrics(val_logits, val_targets, label_mode, label_names)
        score = selection_score(val_metrics)
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'val_macro_f1': score,
                        'val_accuracy': val_metrics['macro']['accuracy'] if label_mode == 'multilabel'
                        else val_metrics['accuracy'],
                        'seconds': round(time.perf_counter() - started, 1)})
        print(f'epoch {epoch:3d}  train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  val_macro_f1 {score:.4f}')
        if best is None or score > best['score']:
            best = {'score': score, 'epoch': epoch, 'metrics': val_metrics}
            torch.save({
                'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'metadata': {
                    'model_name': 'resnet18', 'display_name': 'ResNet18 (fine-tuned)', 'arch': 'resnet18',
                    'num_outputs': num_outputs, 'label_mode': label_mode, 'label_names': label_names,
                    'score_type': 'sigmoid_probability' if label_mode == 'multilabel' else 'softmax_probability',
                    'pretrained': (str(weights_path) if weights_path else PRETRAINED_TAG if pretrained else None),
                    'freeze_backbone': freeze_backbone,
                    'preprocessing': {'size': config.IMAGE_SIZE, 'denoise': denoise, 'channels': 'grayscale x3',
                                      'mean': list(config.IMAGENET_MEAN), 'std': list(config.IMAGENET_STD)},
                    'augmentation': repr(TRAIN_AUGMENTATION),
                    'best_epoch': epoch, 'epochs_planned': epochs, 'selection_metric': 'validation macro-F1',
                    'validation_metrics': val_metrics, 'batch_size': batch_size, 'learning_rate': lr,
                    'optimizer': 'Adam', 'seed': seed, 'device': str(device),
                    'dataset_fingerprint': meta['dataset_fingerprint'], 'n_train': len(train_samples),
                    'n_val': len(val_samples), 'torch_version': str(torch.__version__),
                    'torchvision_version': str(torchvision.__version__),
                    'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                },
            }, out_dir / 'resnet18_best.pt')
        (out_dir / 'resnet18_history.json').write_text(json.dumps(
            {'model': 'resnet18', 'best_epoch': best['epoch'], 'epochs': history}, indent=2))

    checkpoint = torch.load(out_dir / 'resnet18_best.pt', map_location='cpu', weights_only=True)
    (out_dir / 'resnet18_best.json').write_text(json.dumps(checkpoint['metadata'], indent=2))
    return out_dir / 'resnet18_best.pt', history


def main():
    parser = argparse.ArgumentParser(description='Fine-tune ResNet18 on handwriting images.')
    parser.add_argument('--manifest', default=str(config.SPLITS_DIR / 'manifest.csv'))
    parser.add_argument('--out-dir', default=str(config.DEEP_DIR))
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--no-pretrained', action='store_true', help='Start from random weights (not recommended).')
    parser.add_argument('--weights-path', help='Local ImageNet ResNet18 state dict (offline alternative to download).')
    parser.add_argument('--freeze-backbone', action='store_true', help='Train only the new 512->N head.')
    parser.add_argument('--no-denoise', action='store_true')
    parser.add_argument('--seed', type=int, default=config.SEED)
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda or mps')
    args = parser.parse_args()
    if not Path(args.manifest).exists():
        raise SystemExit(f'error: split manifest not found at {args.manifest}. '
                         'Create it first with: python -m src.data.dataset --data-root <dataset folder>')
    path, _ = train(args.manifest, args.out_dir, args.epochs, args.batch_size, args.lr, not args.no_pretrained,
                    args.weights_path, args.freeze_backbone, not args.no_denoise, args.seed, args.device)
    print(f'Best checkpoint: {path}')


if __name__ == '__main__':
    main()
