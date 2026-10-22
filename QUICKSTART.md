# Quick start

```bash
# 1. Install (Python 3.10)
python3.10 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

# 2. Check that the legacy model weights are present (about 104 MB, not stored in git)
ls best_model.h5

# 3. Run
python app.py
```

Open **http://127.0.0.1:5001**. Startup log, when no new models are trained yet:

```text
WARNING ...: Model resnet18 unavailable: Model artifact not found: resnet18_best.pt. Train it first (see README).
WARNING ...: Model svm unavailable: ...
WARNING ...: Model random_forest unavailable: ...
INFO ...: Loaded model legacy_resnet50
 * Running on http://127.0.0.1:5001
```

## Check it works

```bash
curl -F file=@1.1.jpg http://127.0.0.1:5001/api/predict
curl http://127.0.0.1:5001/api/health
python -m pytest -q
```

## Using the web page

- **Analyze:** choose or drop a JPG, PNG or WEBP image (at least 128×128 px, at most 16MB), then click
  *Analyze handwriting*. You get the primary model's output, a comparison of all models, and *Download report*.
- **Dashboard:** model status, recorded legacy results (labelled as recorded), dataset status, and the 85% target.
- **Evaluation:** test metrics, confusion matrices and loss curves once models are trained, plus the legacy
  notebook's recorded history.
- **History:** results from this browser tab only.

To train the SVM, Random Forest and ResNet18 models you need a labelled dataset; see `docs/DATASET.md`
and the "Train and evaluate" section of `README.md`.

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Activate the virtualenv and run `pip install -r requirements.txt` |
| Port 5001 busy | `SCRIPTINSIGHT_PORT=5050 python app.py` |
| `ImportError ... _spropack ... __thread_bss` on macOS | `pip install scipy==1.14.1` |
| Old page behaviour after an update | Hard-refresh (Cmd/Ctrl+Shift+R). Assets are versioned, so this should be rare |
