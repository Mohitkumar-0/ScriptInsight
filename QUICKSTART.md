# Quick start (Windows)

```powershell
# 1. Install (once). Needs Python 3.13 from python.org; the download is large (PyTorch + TensorFlow).
py -3.13 -m venv .venv313tf
.\.venv313tf\Scripts\python.exe -m pip install -r requirements-py313.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 2. Check that the legacy model weights are present (about 104 MB, not stored in git)
dir best_model.h5

# 3. Run
.\run.bat
```

`scripts\add_legacy_tensorflow.ps1` does step 1 with a progress display. If you only have the older
Python 3.10 setup, `setup.bat` builds it instead (`requirements.txt`).

Open **http://127.0.0.1:5001**. Startup log with the current models:

```text
WARNING ...: Model resnet18 unavailable: Model artifact not found: resnet18_best.pt. Train it first (see README).
WARNING ...: Model svm unavailable: ...
WARNING ...: Model random_forest unavailable: ...
INFO ...: Loaded model legacy_resnet50
INFO ...: Loaded model hienwrite_resnet18
INFO ...: Loaded model hienwrite_svr
INFO ...: Loaded model hienwrite_rf
 * Running on http://127.0.0.1:5001
```

The three "unavailable" lines are expected: those classifiers have not been trained (no labelled dataset yet).

## Check it works

```powershell
curl.exe -F file=@1.1.jpg http://127.0.0.1:5001/api/predict
curl.exe http://127.0.0.1:5001/api/health
.\.venv313tf\Scripts\python.exe -m pytest -q
```

## Using the web page

- **Analyze:** choose or drop a JPG, PNG or WEBP image (at least 128×128 px, at most 16MB), then click
  *Analyze handwriting*. You get the primary model's output, a comparison of all models, and *Download report*.
- **Dashboard:** model status, recorded legacy results (labelled as recorded), dataset status, and the 85% target.
- **Evaluation:** HiEnWrite test results against the train-mean baseline; classifier confusion matrices and loss
  curves appear once those models are trained. It also shows the legacy notebook's recorded history.
- **History:** results from this browser tab only.

To train the SVM, Random Forest and ResNet18 models you need a labelled dataset; see `docs/DATASET.md`
and the "Train and evaluate" section of `README.md`.

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Use `run.bat`, or call `.\.venv313tf\Scripts\python.exe` directly, so the right environment is used |
| `Model legacy_resnet50 failed to load (ModuleNotFoundError)` | The environment has no TensorFlow. Use `.venv313tf` (see step 1) |
| Port 5001 busy | `$env:SCRIPTINSIGHT_PORT = 5050; .\run.bat` |
| `py -3.13` not found | Install Python 3.13 from python.org, tick "Add python.exe to PATH" and "py launcher" |
| Old page behaviour after an update | Hard-refresh (Ctrl+Shift+R). Assets are versioned, so this should be rare |
| Development-server warning in the log | Use `.\serve.bat` (waitress) instead of `run.bat` |

## HiEnWrite regression models (experimental)

The trained models and their evaluation are already included, so they work without the dataset. Retraining needs
the HiEnWrite dataset on your PC: run the commands in [docs/HIENWRITE.md](docs/HIENWRITE.md#commands) with
`.\.venv313tf\Scripts\python.exe` and your dataset path. After restarting the app, the *Primary model* menu on
the Analyze tab offers the three HiEnWrite models. Their outputs are five numerical scores marked
**Experimental**. Their measured test error is no better than predicting the average; see the Evaluation tab.
