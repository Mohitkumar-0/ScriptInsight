# OPTIONAL: try to get the legacy Keras ResNet50 model working on Python 3.13.
# Builds a SEPARATE environment (.venv313tf) so the working .venv313 is never touched.
#   powershell -ExecutionPolicy Bypass -File scripts\add_legacy_tensorflow.ps1
# TensorFlow 2.21 is reported to support Python 3.10-3.13 on Windows (CPU only). If pip cannot resolve it,
# send me the error; the fallback is Python 3.10 + setup.bat.
$ErrorActionPreference = 'Continue'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$envDir = Join-Path $root '.venv313tf'
$py = Join-Path $envDir 'Scripts\python.exe'
$clock = [Diagnostics.Stopwatch]::StartNew()

function Step($n, $total, $text) {
    Write-Host ''
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host ("[{0}/{1}] {2}   (elapsed {3:mm\:ss})" -f $n, $total, $text, $clock.Elapsed) -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor Cyan
}

Step 1 4 'Creating the virtual environment (.venv313tf)'
if (-not (Test-Path $py)) { py -3.13 -m venv $envDir }
if (-not (Test-Path $py)) { Write-Host 'Could not create the environment. Is Python 3.13 installed?' -ForegroundColor Red; exit 1 }

Step 2 4 'Upgrading pip'
& $py -m pip install --upgrade pip --progress-bar on

Step 3 4 'Installing project packages (PyTorch is about 200 MB; a live progress bar is shown per file)'
& $py -m pip install -r requirements-py313.txt --extra-index-url https://download.pytorch.org/whl/cpu --progress-bar on
if ($LASTEXITCODE -ne 0) { Write-Host 'Package installation failed. Send me the last 40 lines above.' -ForegroundColor Red; exit 1 }

Step 4 4 'Installing TensorFlow (about 300-600 MB with its dependencies; progress bar per file)'
& $py -m pip install tensorflow --progress-bar on
if ($LASTEXITCODE -ne 0) { Write-Host 'TensorFlow installation failed. Send me the last 40 lines above.' -ForegroundColor Red; exit 1 }

Write-Host ''
Write-Host 'Checking the installation...' -ForegroundColor Cyan
& $py -c "import tensorflow as tf, numpy; print('tensorflow', tf.__version__, 'numpy', numpy.__version__)"
Write-Host ''
Write-Host ("Finished in {0:mm\:ss}." -f $clock.Elapsed) -ForegroundColor Green
Write-Host 'If a version line printed above, run the app with:  .\run.bat'
Write-Host 'and check http://127.0.0.1:5001/api/health lists legacy_resnet50 as available.'
