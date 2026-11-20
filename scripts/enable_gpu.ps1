# Replaces the CPU-only PyTorch in .venv313tf with the CUDA 12.6 build so ResNet18 can train on the NVIDIA GPU.
#   powershell -ExecutionPolicy Bypass -File scripts\enable_gpu.ps1
# The CUDA build also runs on the CPU, so the app and everything else keep working in the same environment.
# Download is large (about 2-3 GB). Close any running training first.
$ErrorActionPreference = 'Continue'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$py = Join-Path $root '.venv313tf\Scripts\python.exe'
if (-not (Test-Path $py)) { Write-Host 'Environment .venv313tf not found.' -ForegroundColor Red; exit 1 }

Write-Host '[1/4] Checking the NVIDIA driver' -ForegroundColor Cyan
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Write-Host 'nvidia-smi was not found. Install the current NVIDIA driver from nvidia.com/Download, restart, and run this again.' -ForegroundColor Red
    exit 1
}
$line = (& nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | Select-Object -First 1)
Write-Host "      GPU and driver: $line"
$driver = [version](($line -split ',')[1].Trim())
if ($driver.Major -lt 561) {
    Write-Host "Driver $driver is too old for the CUDA 12.6 build (needs 561 or newer). Update it from nvidia.com/Download and run this again." -ForegroundColor Red
    exit 1
}

Write-Host '[2/4] Saving the current package list (so the change can be undone)' -ForegroundColor Cyan
& $py -m pip freeze | Out-File -FilePath (Join-Path $root '.venv313tf\pip-freeze-before-gpu.txt') -Encoding utf8

Write-Host '[3/4] Installing the CUDA build of PyTorch (large download, progress bar per file)' -ForegroundColor Cyan
& $py -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126 --force-reinstall --no-deps --progress-bar on
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Install failed. Send me the last 30 lines above. To go back to the CPU build:' -ForegroundColor Red
    Write-Host '  .\.venv313tf\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps'
    exit 1
}

Write-Host '[4/4] Checking that PyTorch sees the GPU' -ForegroundColor Cyan
& $py -c "import torch; print('torch', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); print('vram GB:', round(torch.cuda.get_device_properties(0).total_memory/1e9,1) if torch.cuda.is_available() else 0)"
Write-Host ''
Write-Host 'If it says "cuda available: True", train with:' -ForegroundColor Green
Write-Host '  .\.venv313tf\Scripts\python.exe -m src.deep_learning.train_resnet18 --epochs 10 --freeze-backbone --device cuda'
