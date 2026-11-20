"""Fine-tune torchvision ResNet18 with a five-output regression head on HiEnWrite (experimental).

* Head: the 1000-class ImageNet layer is replaced by Linear(512 -> 5), one output per trait. Outputs are
  standardised scores (z = (score - train mean) / train std); inference converts them back to the raw scale.
* Loss: masked MSE (or SmoothL1) over valid targets only. Openness is masked for Hindi rows.
* Preprocessing: grayscale, 3x3 Gaussian denoise, aspect-preserving resize onto a square of the page's median
  gray (``fit='pad'``), 3 identical channels, ImageNet mean/std. Inference uses the same function
  (``prepare_gray``) with the parameters stored in the checkpoint.
* Augmentation (train split only): small RandomAffine (rotation <= 10 deg, translate 3%, scale 0.95-1.05,
  shear 5 deg) and brightness/contrast jitter for phone-camera lighting. No flips: mirrored script is not
  a realistic sample.
* Checkpoint selection and early stopping use the validation loss; the test split is not read.
* Device: CUDA, then MPS after a CPU/MPS agreement check, otherwise CPU.

Usage:
    python -m src.deep_learning.train_resnet18_regression --smoke          # quick pipeline check
    python -m src.deep_learning.train_resnet18_regression --epochs 40
"""
import argparse
import json
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
from src.data.hienwrite import TARGET_COLUMNS, TARGETS, read_split_manifest, stack_targets, target_stats
from src.deep_learning.model import PRETRAINED_TAG, build_resnet18
from src.deep_learning.train_resnet18 import TRAIN_AUGMENTATION as AFFINE_AUGMENTATION
from src.deep_learning.train_resnet18 import seed_everything
from src.evaluation.metrics import normalized_rmse, regression_metrics
from src.preprocessing.image_processor import decode_image_file, prepare_gray, resnet18_tensor

MODEL_NAME = 'hienwrite_resnet18'
DISPLAY_NAME = 'HiEnWrite ResNet18 regression · experimental'
TRAIN_AUGMENTATION = transforms.Compose([AFFINE_AUGMENTATION, transforms.ColorJitter(brightness=0.2, contrast=0.2)])
SMOKE_DIR = config.HIENWRITE_ARTIFACTS_DIR / 'smoke'


def preprocessing(image_size, denoise=True):
    return {'size': image_size, 'denoise': denoise, 'fit': 'pad', 'channels': 'grayscale x3',
            'mean': list(config.IMAGENET_MEAN), 'std': list(config.IMAGENET_STD)}


def _prepare(path, prep):
    return prepare_gray(decode_image_file(path, min_side=1).gray, prep)


class RegressionDataset(Dataset):
    """Images are decoded and preprocessed once; only the random augmentation runs per epoch."""

    def __init__(self, samples, prep, target_mean, target_std, augment, n_jobs=-1):
        from joblib import Parallel, delayed
        self.images = Parallel(n_jobs=n_jobs)(delayed(_prepare)(s['path'], prep) for s in samples)
        Y, M = stack_targets(samples)
        self.z = np.nan_to_num((Y - target_mean) / target_std).astype(np.float32)
        self.mask = M.astype(np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.augment:
            image = np.asarray(TRAIN_AUGMENTATION(Image.fromarray(image)), dtype=np.uint8)
        return torch.from_numpy(resnet18_tensor(image)), torch.from_numpy(self.z[idx]), torch.from_numpy(self.mask[idx])


def masked_loss(pred, target, mask, kind='mse'):
    if kind == 'smoothl1':
        per_value = nn.functional.smooth_l1_loss(pred, target, reduction='none')
    else:
        per_value = (pred - target) ** 2
    return (per_value * mask).sum() / mask.sum().clamp(min=1)


def pick_device(name='auto'):
    """Returns (device, message). MPS is used only if it reproduces CPU outputs on a test batch."""
    if name != 'auto':
        return torch.device(name), f'device {name} (requested)'
    if torch.cuda.is_available():
        return torch.device('cuda'), 'device cuda'
    if torch.backends.mps.is_available():
        try:
            torch.manual_seed(0)
            model = build_resnet18(len(TARGETS)).eval()
            x = torch.randn(2, 3, 224, 224)
            with torch.no_grad():
                ref = model(x)
                out = model.to('mps')(x.to('mps')).cpu()
            if torch.allclose(ref, out, atol=1e-3, rtol=1e-3):
                return torch.device('mps'), 'device mps (Apple GPU; outputs matched CPU on a check batch)'
            return torch.device('cpu'), 'MPS outputs differed from CPU on a check batch; falling back to CPU'
        except Exception as exc:  # MPS present but unusable
            return torch.device('cpu'), f'MPS unavailable ({exc.__class__.__name__}); falling back to CPU'
    return torch.device('cpu'), 'device cpu (no CUDA or MPS)'


def run_epoch(model, loader, device, loss_kind, optimizer=None):
    model.train(optimizer is not None)
    total, weight, outputs = 0.0, 0.0, []
    with torch.set_grad_enabled(optimizer is not None):
        for x, z, m in loader:
            x, z, m = x.to(device), z.to(device), m.to(device)
            pred = model(x)
            loss = masked_loss(pred, z, m, loss_kind)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += float(loss.detach()) * float(m.sum())
            weight += float(m.sum())
            outputs.append(pred.detach().cpu())
    return total / max(weight, 1.0), torch.cat(outputs).numpy()


def train(manifest=config.HIENWRITE_SPLITS_DIR / 'split_manifest.csv', out_dir=config.HIENWRITE_MODELS_DIR,
          epochs=40, batch_size=16, lr=1e-4, weight_decay=1e-4, image_size=224, pretrained=True, weights_path=None,
          freeze_backbone=False, loss_kind='mse', patience=8, seed=config.SEED, device='auto', num_workers=0,
          data_root=None, max_samples=None, n_jobs=-1, log=print):
    seed_everything(seed)
    samples, meta = read_split_manifest(manifest, data_root)
    train_samples = [s for s in samples if s['split'] == 'train']
    val_samples = [s for s in samples if s['split'] == 'val']
    if max_samples:  # smoke test only
        train_samples, val_samples = train_samples[:max_samples], val_samples[:max(4, max_samples // 2)]
    device, message = pick_device(device)
    log(message)

    stats = target_stats(train_samples)
    mean = np.array([stats[t]['mean'] for t in TARGETS])
    std = np.array([stats[t]['std'] for t in TARGETS])
    prep = preprocessing(image_size)
    started_all = time.perf_counter()
    train_ds = RegressionDataset(train_samples, prep, mean, std, augment=True, n_jobs=n_jobs)
    val_ds = RegressionDataset(val_samples, prep, mean, std, augment=False, n_jobs=n_jobs)
    log(f'loaded {len(train_ds)} train / {len(val_ds)} val images in {time.perf_counter() - started_all:.1f}s')
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)

    model = build_resnet18(len(TARGETS), pretrained=pretrained, weights_path=weights_path,
                           freeze_backbone=freeze_backbone).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    Y_val, M_val = stack_targets(val_samples)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / config.HIENWRITE_RESNET18_PATH.name
    history_path = out_dir / config.HIENWRITE_RESNET18_HISTORY_PATH.name
    history, best, stale = [], None, 0
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_loss, _ = run_epoch(model, train_loader, device, loss_kind, optimizer)
        val_loss, val_z = run_epoch(model, val_loader, device, loss_kind)
        scheduler.step(val_loss)
        val_metrics = regression_metrics(Y_val, val_z * std + mean, TARGETS, M_val)
        val_metrics['macro']['nrmse'] = normalized_rmse(val_metrics, {t: stats[t]['std'] for t in TARGETS})
        entry = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                 'val_macro_rmse': val_metrics['macro']['rmse'], 'val_macro_mae': val_metrics['macro']['mae'],
                 'val_macro_pearson_r': val_metrics['macro']['pearson_r'],
                 'lr': optimizer.param_groups[0]['lr'], 'seconds': round(time.perf_counter() - started, 1)}
        history.append(entry)
        r = entry['val_macro_pearson_r']
        log(f"epoch {epoch:3d}  train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  val_RMSE {entry['val_macro_rmse']:.3f}"
            f"  val_r {'n/a' if r is None else f'{r:+.3f}'}  ({entry['seconds']}s)")
        if best is None or val_loss < best['val_loss']:
            best, stale = {'val_loss': val_loss, 'epoch': epoch}, 0
            torch.save({
                'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'metadata': {
                    'model_name': MODEL_NAME, 'display_name': DISPLAY_NAME, 'arch': 'resnet18',
                    'dataset': 'HiEnWrite', 'experimental': True,
                    'num_outputs': len(TARGETS), 'label_mode': 'regression', 'task': 'regression',
                    'label_names': list(TARGETS), 'csv_columns': TARGET_COLUMNS, 'score_type': 'regression_score',
                    'target_mean': mean.tolist(), 'target_std': std.tolist(), 'target_stats': stats,
                    'preprocessing': prep, 'augmentation': repr(TRAIN_AUGMENTATION),
                    'pretrained': (str(weights_path) if weights_path else PRETRAINED_TAG if pretrained else None),
                    'freeze_backbone': freeze_backbone, 'loss': f'masked {loss_kind} on standardised targets',
                    'best_epoch': epoch, 'epochs_planned': epochs, 'early_stopping_patience': patience,
                    'selection_metric': 'validation loss (masked, standardised targets)',
                    'validation_loss': val_loss, 'validation_metrics': val_metrics,
                    'batch_size': batch_size, 'learning_rate': lr, 'weight_decay': weight_decay,
                    'optimizer': 'AdamW + ReduceLROnPlateau(factor 0.5, patience 3)', 'seed': seed, 'device': str(device),
                    'dataset_fingerprint': meta['dataset_fingerprint'], 'split_seed': meta['seed'],
                    'n_train': len(train_samples), 'n_val': len(val_samples), 'smoke_test': bool(max_samples),
                    'torch_version': str(torch.__version__), 'torchvision_version': str(torchvision.__version__),
                    'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                },
            }, checkpoint_path)
        else:
            stale += 1
        history_path.write_text(json.dumps({'model': MODEL_NAME, 'best_epoch': best['epoch'], 'device': str(device),
                                            'loss': f'masked {loss_kind} on standardised targets',
                                            'epochs': history}, indent=2))
        if patience and stale >= patience:
            log(f'early stopping: validation loss has not improved for {patience} epochs (best epoch {best["epoch"]})')
            break

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    checkpoint['metadata']['epochs_run'] = len(history)
    checkpoint['metadata']['training_seconds'] = round(time.perf_counter() - started_all, 1)
    torch.save(checkpoint, checkpoint_path)
    checkpoint_path.with_suffix('.json').write_text(json.dumps(checkpoint['metadata'], indent=2))
    return checkpoint_path, history


def main():
    parser = argparse.ArgumentParser(description='Fine-tune ResNet18 for five-trait regression on HiEnWrite.')
    parser.add_argument('--manifest', default=str(config.HIENWRITE_SPLITS_DIR / 'split_manifest.csv'))
    parser.add_argument('--data-root', help='Dataset folder, if it moved since the manifest was written.')
    parser.add_argument('--out-dir', help=f'Default {config.HIENWRITE_MODELS_DIR} ({SMOKE_DIR} with --smoke).')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--loss', choices=['mse', 'smoothl1'], default='mse')
    parser.add_argument('--patience', type=int, default=8, help='Early-stopping patience in epochs (0 = off).')
    parser.add_argument('--no-pretrained', action='store_true', help='Start from random weights (not recommended).')
    parser.add_argument('--weights-path', help='Local ImageNet ResNet18 state dict (offline alternative to download).')
    parser.add_argument('--freeze-backbone', action='store_true', help='Train only the 512->5 head.')
    parser.add_argument('--seed', type=int, default=config.SEED)
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda or mps')
    parser.add_argument('--workers', type=int, default=0, help='DataLoader worker processes.')
    parser.add_argument('--smoke', action='store_true',
                        help='Pipeline check: 32 train images, 1 epoch, written to the smoke directory.')
    args = parser.parse_args()
    out_dir = args.out_dir or (SMOKE_DIR if args.smoke else config.HIENWRITE_MODELS_DIR)
    try:
        path, history = train(args.manifest, out_dir, 1 if args.smoke else args.epochs, args.batch_size, args.lr,
                              args.weight_decay, args.image_size, not args.no_pretrained, args.weights_path,
                              args.freeze_backbone, args.loss, args.patience, args.seed, args.device, args.workers,
                              args.data_root, 32 if args.smoke else None)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f'error: {exc}')
    print(f'Best checkpoint: {path}')
    if args.smoke:
        from src.deep_learning.predict_resnet18 import ResNet18Model
        samples, _ = read_split_manifest(args.manifest, args.data_root)
        model = ResNet18Model(path)
        out = model.predict_gray(decode_image_file(samples[0]['path'], min_side=1).gray)
        print(f'smoke test: checkpoint reloaded; prediction for {samples[0]["image"]}: '
              + ', '.join(f'{k} {v:.2f}' for k, v in out['scores'].items()))


if __name__ == '__main__':
    main()
