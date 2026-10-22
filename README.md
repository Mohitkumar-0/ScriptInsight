# ScriptInsight: Intelligent Personality Prediction using Handwriting Analysis

An **experimental** research prototype. A Flask web app takes a handwriting image and runs every
available model on it: HOG+LBP **SVM**, HOG+LBP **Random Forest**, fine-tuned **ResNet18**, plus a
legacy Keras ResNet50 demo. It shows their outputs side by side, together with the evaluation evidence.

> **Disclaimer.** Inferring personality from handwriting is not a validated psychological assessment.
> Do not use the output for hiring, clinical, educational or other consequential decisions.

## Current status (what is real today)

| Component | Status |
|---|---|
| Upload, validation, preview, drag-and-drop, history, report export | Working, tested (pytest + Chrome/WebKit E2E) |
| Legacy ResNet50 (`best_model.h5`) | Working. **Single-label, 5-category** classifier; recorded test accuracy **30.3%** (chance ≈ 20%) |
| HOG + LBP feature extraction | Implemented and tested |
| SVM, Random Forest, ResNet18 | Code implemented and tested end-to-end on synthetic data. **Not trained**: no labelled dataset is available (see [docs/DATASET_DISCOVERY.md](docs/DATASET_DISCOVERY.md)) |
| Model comparison / evaluation / loss charts | Implemented. Show real data when models are trained; until then they show the legacy model's *recorded* notebook results, labelled as such |
| 85% validation-accuracy target (Openness, Conscientiousness) | **Not met / not measurable.** See [Limitations](#limitations) |

## Project structure

```text
app.py                         Flask app factory and routes
config.py                      All settings (env vars SCRIPTINSIGHT_*)
src/
  preprocessing/image_processor.py   decode + validate uploads, grayscale, 224x224, denoise, tensors
  features/handwriting_features.py   HOG + LBP (6262-dim, versioned)
  data/dataset.py                    dataset loading, leakage-aware 80:10:10 split, manifest
  data/hwxpi.py                      converter for the ChaLearn HWxPI dataset
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
artifacts/legacy_resnet50/recorded_metrics.json   legacy results extracted from the notebook
artifacts/splits, artifacts/evaluation             split manifests and evaluation outputs
scripts/extract_legacy_records.py, scripts/e2e_ui.cjs
tests/                         pytest suite
docs/                          dataset guide, dataset discovery report, report-correction checklist
personality_prediction_resnet.ipynb   original training notebook (legacy model)
```

## Setup

Requires **Python 3.10** (3.8-3.11 work with TensorFlow 2.13). CPU only by default.

```bash
python3.10 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt  # runtime + pytest
```

Pip will warn that `tensorflow 2.13.1 requires typing-extensions<4.6`. This is expected: PyTorch 2.2
needs a newer version, and TensorFlow works with it (the test suite covers the legacy model).

**Platforms.** Tested on macOS (Apple M4, Python 3.10.22). Dependency resolution was checked for
Linux x86-64 and Windows x64 (all wheels exist), but the app has **not been run** on those systems.

**GPU (optional).** Inference always runs on CPU (fast enough: about 50 ms per image for the legacy model).
For faster ResNet18 training, install a CUDA build of `torch`/`torchvision` 2.2.2/0.17.2 from pytorch.org
and pass `--device cuda`; on Apple silicon `--device mps` works. Do not install `tensorflow-metal`:
it corrupted the legacy model's outputs.

## Run the app

```bash
python app.py                        # http://127.0.0.1:5001
```

Settings come from environment variables: `SCRIPTINSIGHT_HOST` (default `127.0.0.1`; use
`0.0.0.0` only if you need LAN access), `SCRIPTINSIGHT_PORT` (5001), `SCRIPTINSIGHT_MAX_UPLOAD_MB` (16),
`SCRIPTINSIGHT_MIN_IMAGE_SIDE` (128), `SCRIPTINSIGHT_ENABLE_LEGACY_MODEL` (true), `SCRIPTINSIGHT_DENOISE` (true).
For production, use a WSGI server, e.g. `gunicorn "app:create_app()"` (Linux/macOS).

## Train and evaluate (needs a labelled dataset)

See [docs/DATASET.md](docs/DATASET.md) for the expected layout and for how to obtain HWxPI.

```bash
# 1. leakage-aware 80:10:10 split (grouped by participant, stratified where possible)
python -m src.data.dataset --data-root data/hwxpi

# 2. classical models (HOG + LBP, StandardScaler fitted on train, hyper-parameters chosen on validation)
python -m src.classical.train_classical

# 3. ResNet18 (downloads ImageNet weights on first run, about 45 MB, to ~/.cache/torch)
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
| GET | `/api/metrics` | latest evaluation, split metadata, training histories, legacy recorded results |

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
Errors return `{"success": false, "error": "...", "code": "..."}` with status 400 (no or empty file),
413 (too large), 415 (unsupported type), 422 (corrupt or too small), 503 (no model) or 500.

## Testing

```bash
python -m pytest -q                           # 55 tests: preprocessing, features, split, metrics, models, API, pipeline
# Browser E2E (optional, needs Node 18+ and playwright-core; see header of scripts/e2e_ui.cjs)
NODE_PATH=/path/to/node_modules node scripts/e2e_ui.cjs http://127.0.0.1:5001 chrome
```

## Privacy

Uploaded images are decoded in memory and never written to disk. The History tab stores result
summaries (never images) in the browser tab's `sessionStorage`; it can be switched off or cleared, and it
is discarded when the tab closes.

## Limitations

- **No labelled dataset is available**, so SVM, Random Forest and ResNet18 are untrained, and the report's
  comparative evaluation and 85% target cannot be measured yet.
- The legacy model answers "which of five categories", not five independent trait levels. Its recorded
  results (30.3% test accuracy) come from the notebook and cannot be re-run: its dataset is missing and has
  no documented label source. The notebook also shows a probable train/validation leakage risk and a
  training-loop defect (every other epoch trained on one batch).
- Published work on the HWxPI benchmark reports near-chance results (best challenge AUC about 0.53), so high
  accuracy should not be expected from handwriting alone.

Older notes (`IMPLEMENTATION_COMPLETE.md`, `THEME_*.md`, `DARK_LIGHT_MODE.md`, `PERSONALITY_DETAILS.md`)
describe the earlier UI and are kept for history only.
