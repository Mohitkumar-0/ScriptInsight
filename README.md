# ScriptInsight: Intelligent Personality Prediction using Handwriting Analysis

An **experimental** research prototype. A Flask web app takes a handwriting image and runs every
available model on it: HOG+LBP **SVM**, HOG+LBP **Random Forest**, fine-tuned **ResNet18**, plus a
legacy Keras ResNet50 demo. It shows their outputs side by side, together with the evaluation evidence.

It also includes **experimental five-trait regression models trained on the HiEnWrite dataset**
(Checker, Yadav & Katarya 2025, CC BY-NC 4.0, non-commercial use only): SVR (HOG), Random Forest
(HOG + LBP) and ResNet18. They predict five numerical questionnaire scores. **On the held-out test split
they do not beat predicting the training mean.** See [docs/HIENWRITE.md](docs/HIENWRITE.md).

> **Disclaimer.** Inferring personality from handwriting is not a validated psychological assessment.
> Do not use the output for hiring, clinical, educational or other consequential decisions.

## Current status (what is real today)

| Component | Status |
|---|---|
| Upload, validation, preview, drag-and-drop, history, report export | Working, tested (pytest + Chrome/WebKit E2E) |
| Legacy ResNet50 (`best_model.h5`) | Working. **Single-label, 5-category** classifier; recorded test accuracy **30.3%** (chance ≈ 20%) |
| HOG + LBP feature extraction | Implemented and tested |
| SVM, Random Forest, ResNet18 (5-category classifiers) | **Trained as DEMO models** (18 Nov 2026) on the Kaggle handwriting set: 4,076 augmented images from 221 original photos, split by source photo (3,260 / 408 / 408). Test accuracy **32.8% / 37.8% / 27.2%**, macro-F1 0.32 / 0.39 / 0.26, against a 25.5% majority-class baseline and 20% chance. The test split holds only about 22 independent photos (one photo is about 4.5 points), the labels have no documented source, each category was photographed in one session and writers overlap, so these numbers do **not** show that handwriting predicts personality. See [docs/KAGGLE_DEMO_DATASET.md](docs/KAGGLE_DEMO_DATASET.md) |
| Model comparison / evaluation / loss charts | Implemented. Show real data when models are trained; until then they show the legacy model's *recorded* notebook results, labelled as such |
| 85% validation-accuracy target (Openness, Conscientiousness) | **Not met / not measurable.** See [Limitations](#limitations) |
| **HiEnWrite regression** (SVR, Random Forest, ResNet18; five numerical scores) | **Trained** on 405 images, model choice on 87, **tested on 86** (participant-grouped split). Test macro RMSE 5.40 / 5.41 / 5.87 vs **5.24 for the train-mean baseline**; Pearson r ≈ 0. Served in the app as *experimental*; [docs/HIENWRITE.md](docs/HIENWRITE.md) |
| Legacy model under Python 3.13 | **Working** on Windows 11 with Python 3.13.3 and TensorFlow 2.21.0 (Keras 3.15.1); verified 17 Nov 2026. In an environment without TensorFlow the app reports a load error for this one model and keeps running |

## Project structure

```text
app.py                         Flask app factory and routes
config.py                      All settings (env vars SCRIPTINSIGHT_*)
run.bat, serve.bat, setup.bat  Windows launchers: dev server, waitress server, Python 3.10 environment setup
requirements.txt               Python 3.10 pins (TensorFlow 2.13); requirements-py313.txt: Python 3.13 pins (TensorFlow 2.21)
PROGRESS.md                    Working log of the completion work (what was changed, tested, still open)
src/
  preprocessing/handwriting_validator.py   rejects blank pages, faces and non-handwriting uploads
  preprocessing/image_processor.py   decode + validate uploads, grayscale, 224x224, denoise, tensors
  features/handwriting_features.py   HOG + LBP (6262-dim, versioned)
  data/dataset.py                    dataset loading, leakage-aware 80:10:10 split, manifest
  data/hwxpi.py                      converter for the ChaLearn HWxPI dataset
  data/hienwrite.py                  HiEnWrite validation, data manifest, participant-grouped split
  classical/train_regression.py      HiEnWrite SVR / Random Forest regressors
  deep_learning/train_resnet18_regression.py   HiEnWrite ResNet18 five-output regression
  evaluation/evaluate_regression.py  HiEnWrite test-split evaluation vs train-mean baseline
  classical/train_classical.py       SVM + Random Forest training (validation-split model selection)
  classical/predict_classical.py     inference for saved classical bundles
  deep_learning/model.py             torchvision ResNet18 with 512 -> N head
  deep_learning/train_resnet18.py    fine-tuning with train-only augmentation
  deep_learning/predict_resnet18.py  checkpoint inference (eval mode, no grad)
  deep_learning/legacy_resnet50.py   legacy Keras ResNet50 loader
  evaluation/metrics.py              metric definitions
  evaluation/evaluate.py             unified test-split evaluation -> JSON + PNG
  services/prediction_service.py     loads models once, runs predictions
templates/index.html, static/       Flask template + vanilla JS (Analyze, Dashboard, Evaluation, History)
models/classical, models/deep_learning   trained artifacts (created by training; git-ignored)
models/hienwrite                    HiEnWrite regression artifacts (weights git-ignored, .json metadata kept)
artifacts/hienwrite                 HiEnWrite validation report, manifests, splits, evaluation
artifacts/legacy_resnet50/recorded_metrics.json   legacy results extracted from the notebook
artifacts/splits, artifacts/evaluation             split manifests and evaluation outputs
scripts/                       extract_legacy_records.py, e2e_ui.cjs, phase0_cleanup.ps1, phase2_baseline.ps1, add_legacy_tensorflow.ps1
tests/                         pytest suite
docs/                          dataset guide, dataset discovery report, handwriting validation, report-correction checklist
docs/archive/                  superseded notes from the earlier UI (kept for history)
personality_prediction_resnet.ipynb   original training notebook (legacy model)
```

## Setup (Windows)

Two environments are supported; both are CPU-only by default.

| Environment | Python | Packages | Notes |
|---|---|---|---|
| **Recommended** | 3.13 | `requirements-py313.txt` | Everything works, including the legacy ResNet50 (TensorFlow 2.21) |
| Original | 3.10 | `requirements.txt` | Same features with TensorFlow 2.13. `setup.bat` builds it (needs Python 3.10 installed) |

Recommended setup (PowerShell, from the project folder):

```powershell
py -3.13 -m venv .venv313tf
.\.venv313tf\Scripts\python.exe -m pip install -r requirements-py313.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

The download is large (PyTorch and TensorFlow, roughly 1 GB). `scripts\add_legacy_tensorflow.ps1` does the same
with a step-by-step progress display. `run.bat` and `serve.bat` pick the first environment they find in this
order: `.venv`, `.venv313tf`, `.venv313`.

**Platforms.** Verified on Windows 11 with Python 3.13.3 (18 Nov 2026: 130 tests pass, app and all seven
available models run). Earlier development was done on macOS (Apple M4). Linux has not been run.

**Version note.** Trained `.joblib` files only load with the numpy/scikit-learn versions that saved them
(scikit-learn 1.9.1 for the HiEnWrite SVR and Random Forest); `requirements-py313.txt` matches. The
`requirements.txt` pins (numpy 1.24.3, scikit-learn 1.3.2) belong to the Python 3.10 environment and cannot be
installed on 3.13. In the 3.10 environment pip prints a harmless warning that `tensorflow 2.13.1 requires
typing-extensions<4.6`; PyTorch 2.2 needs a newer version and the test suite shows they work together.

**GPU (optional).** Inference always runs on CPU (about 50-110 ms per model and image). TensorFlow has no GPU
support on native Windows. For faster ResNet18 training, `scripts\enable_gpu.ps1` replaces the CPU-only PyTorch in
`.venv313tf` with the CUDA 12.6 build (about 2.6 GB download; needs NVIDIA driver 561 or newer). Verified on an
RTX 3050 Laptop GPU (4 GB, driver 610.47, torch 2.14.1+cu126). Then pass `--device cuda` to the ResNet18 trainers.
The CUDA build also runs on the CPU, so the app is unaffected. The SVM and Random Forest always train on the CPU.

## Run the app

```powershell
.\run.bat          # development server, http://127.0.0.1:5001
.\serve.bat        # waitress server (use this instead of the development server for anything beyond local testing)
```

Or directly: `.\.venv313tf\Scripts\python.exe app.py`.

Settings come from environment variables: `SCRIPTINSIGHT_HOST` (default `127.0.0.1`; use
`0.0.0.0` only if you need LAN access), `SCRIPTINSIGHT_PORT` (5001), `SCRIPTINSIGHT_MAX_UPLOAD_MB` (16),
`SCRIPTINSIGHT_MIN_IMAGE_SIDE` (128), `SCRIPTINSIGHT_ENABLE_LEGACY_MODEL` (true), `SCRIPTINSIGHT_DENOISE` (true).
`gunicorn` does not run on Windows; `serve.bat` uses `waitress` (`python -m waitress --call app:create_app`).

Models that have not been trained (ResNet18, SVM and Random Forest classifiers) show "Train it first" in the app;
that is expected until a labelled dataset is available (see below).

## HiEnWrite regression: train and evaluate

Full details, findings and results: [docs/HIENWRITE.md](docs/HIENWRITE.md).

```powershell
$PY = ".\.venv313tf\Scripts\python.exe"
& $PY -m src.data.hienwrite validate --data-root C:\path\to\HiEnWrite-Dataset-main
& $PY -m src.data.hienwrite split --seed 42
& $PY -m src.classical.train_regression
& $PY -m src.deep_learning.train_resnet18_regression --smoke
& $PY -m src.deep_learning.train_resnet18_regression --epochs 60 --freeze-backbone --lr 1e-3
& $PY -m src.evaluation.evaluate_regression
```

The dataset is **not** included in the repository (download it from the HiEnWrite GitHub project). The manifests
already in `artifacts/hienwrite/` record the dataset location of the machine that created them. After moving
to another PC, re-run the `validate` and `split` lines with your own path to regenerate them (the dataset
fingerprint `efe899f06db72c1d` shows whether the data and split are identical), or pass `--data-root`
where a script offers it. The trained models and the evaluation results are in the project folder (the weights are
git-ignored, so back them up separately), so the app runs without the dataset.

## Train and evaluate the classifiers (needs a labelled dataset)

See [docs/DATASET.md](docs/DATASET.md) for the expected layout and for how to obtain HWxPI.

```bash
# 1. leakage-aware 80:10:10 split (grouped by participant, stratified where possible)
python -m src.data.dataset --data-root data/hwxpi

# 2. classical models (HOG + LBP, StandardScaler fitted on train, hyper-parameters chosen on validation)
python -m src.classical.train_classical

# 3. ResNet18 (downloads ImageNet weights on first run, about 45 MB, to %USERPROFILE%\.cache\torch)
python -m src.deep_learning.train_resnet18 --epochs 25
#    offline: --weights-path resnet18-f37072fd.pth   |   head only: --freeze-backbone

# 4. evaluate every trained model on the same held-out test split
python -m src.evaluation.evaluate
```

Restart the app afterwards: it loads models once at startup. Outputs:
- `models/classical/{svm,random_forest}.joblib` plus `.json` metadata
- `models/deep_learning/resnet18_best.pt`, `resnet18_best.json`, `resnet18_history.json`
- `artifacts/splits/manifest.csv` and `split_meta.json`
- `artifacts/evaluation/latest.json` and confusion-matrix / loss PNGs

`*.joblib` files are pickles: only load artifacts you produced yourself.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/predict` and `/predict` | multipart field `file`; returns all model outputs |
| GET | `/api/info` | traits, model status and metadata, upload limits |
| GET | `/api/health` | `ok` (200) or `degraded` (503, no model available) |
| GET | `/api/metrics` | latest evaluation, split metadata, training histories, legacy recorded results, `hienwrite` block |

`/api/predict` also accepts an optional form field `model` (`resnet18`, `svm`, `random_forest`,
`legacy_resnet50`, `hienwrite_resnet18`, `hienwrite_svr`, `hienwrite_rf`, or `auto`) to choose the primary
result. An unknown name returns 400 `unknown_model`. A model that is not loaded returns 503
`model_unavailable`; no other model is substituted. Without it, the default is unchanged: classifiers
first, and the HiEnWrite regressors only when no classifier is loaded.

Successful prediction response (legacy model only, as today):

```json
{
  "success": true,
  "prediction": {"model": "legacy_resnet50", "task": "multiclass", "predicted_personality": "Openness",
                 "confidence": 0.3678, "probabilities": {"Agreeableness": 0.2464, "...": 0.0},
                 "score_type": "softmax_probability", "display_name": "Legacy ResNet50 (5-category classifier)"},
  "model_predictions": [{"model": "resnet18", "status": "unavailable", "message": "..."}, {"model": "legacy_resnet50", "status": "ok", "scores": {}, "latency_ms": 52.1}],
  "image": {"width": 900, "height": 1600, "format": "JPEG"},
  "model_metadata": {"primary_model": "legacy_resnet50", "trait_order": ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]},
  "warnings": ["Only the legacy 5-category model is available ..."],
  "disclaimer": "Experimental research prototype ..."
}
```

When a multilabel model (trained on five binary trait labels) is primary, `prediction` contains
`"task": "multilabel"` and `"traits": {"Openness": {"probability": 0.62, "level": "high"}, ...}`.
When a HiEnWrite regressor is primary, it contains `"task": "regression"`, `"experimental": true`,
`"trait_scores": {"Openness": 26.1, ...}` (raw questionnaire scale, not probabilities) and
`"training_label_stats"` (training mean/min/max per trait).
Errors return `{"success": false, "error": "...", "code": "..."}` with status 400 (no or empty file),
413 (too large), 415 (unsupported type), 422 (corrupt or too small), 503 (no model) or 500.

## Testing

```powershell
.\.venv313tf\Scripts\python.exe -m pytest -q   # 130 passed (Windows 11, Python 3.13.3, TensorFlow 2.21.0)
```

Without TensorFlow (for example `.venv313`) the 4 legacy-model tests skip and the rest pass. In the Python 3.10
environment use `.venv\Scripts\python.exe -m pytest -q`; there the trained HiEnWrite pickles need scikit-learn 1.9.1
and their tests skip or the models report "cannot be loaded".

Browser end-to-end test (optional, needs Node 18+ and playwright-core; see the header of `scripts\e2e_ui.cjs`):

```powershell
$env:NODE_PATH = "C:\path\to\node_modules"
node scripts\e2e_ui.cjs http://127.0.0.1:5001 chrome
```

## Privacy

Uploaded images are decoded in memory and never written to disk. The History tab stores result
summaries (never images) in the browser tab's `sessionStorage`; it can be switched off or cleared, and it
is discarded when the tab closes.

## Limitations

- **No dataset with binary or categorical trait labels is available**, so the SVM, Random Forest and
  ResNet18 *classifiers* are untrained, and the report's 85% accuracy target cannot be measured.
- The **HiEnWrite regression** models are trained on a different task: five numerical scores. Their test
  error is worse than predicting the training mean. They are not comparable with the legacy classifier
  and are not HWxPI results. See [docs/HIENWRITE.md](docs/HIENWRITE.md#limitations).
- The legacy model answers "which of five categories", not five independent trait levels. Its recorded
  results (30.3% test accuracy) come from the notebook and cannot be re-run: its dataset is missing and has
  no documented label source. The notebook also shows a probable train/validation leakage risk and a
  training-loop defect (every other epoch trained on one batch).
- Published work on the HWxPI benchmark reports near-chance results (best challenge AUC about 0.53), so high
  accuracy should not be expected from handwriting alone.

Older notes (`IMPLEMENTATION_COMPLETE.md`, `THEME_*.md`, `DARK_LIGHT_MODE.md`) describe the earlier UI and are kept
in [docs/archive](docs/archive) for history only. `PERSONALITY_DETAILS.md` remains in the project root.
