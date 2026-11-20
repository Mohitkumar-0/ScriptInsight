# Changelog

## Unreleased (Windows completion work, 17 November 2026)

### Changed
- SVM, Random Forest and ResNet18 classifiers trained as clearly labelled DEMO models on the Kaggle handwriting
  set (docs/KAGGLE_DEMO_DATASET.md); they are not evidence that handwriting predicts personality.
- Runs on Windows 11. Verified with Python 3.13.3, TensorFlow 2.21.0 (Keras 3.15.1), PyTorch 2.14.1 (CPU),
  scikit-learn 1.9.1: 130 tests pass (18 Nov 2026), all seven models load (legacy ResNet50, three HiEnWrite
  regressors, demo SVM / Random Forest / ResNet18), the web app, history, dark mode, report download, dashboard and evaluation tab were checked by hand.
- All CSV, JSON and Markdown reads and writes specify UTF-8 explicitly (Windows defaults to cp1252).
- `train_resnet18.py`: `float(loss.detach())` removes a PyTorch warning.
- README, QUICKSTART and docs/HIENWRITE.md, docs/DATASET.md rewritten for Windows (macOS paths and the
  Python 3.10-only instructions removed).

### Added
- Nested dataset layout support, `--note` warnings and a majority-class baseline in the evaluation;
  `tests/test_nested_layout.py`; `scripts/enable_gpu.ps1` (CUDA PyTorch, verified on an RTX 3050 Laptop GPU);
  `docs/KAGGLE_DEMO_DATASET.md`.
- `requirements-py313.txt` (pins of the working Python 3.13 environment, including TensorFlow 2.21.0 and waitress).
- `run.bat`, `serve.bat` (waitress), `setup.bat` (Python 3.10 environment).
- `scripts/phase0_cleanup.ps1`, `scripts/phase2_baseline.ps1`, `scripts/add_legacy_tensorflow.ps1`.
- `tests/test_load_json.py`, `PROGRESS.md`, `PLAN_SUMMARY.md`.
- waitress in `requirements.txt`.

### Removed / moved
- macOS metadata files (`._*`, `.DS_Store`) and macOS virtual environments deleted; `.gitignore` now ignores them.
- Superseded notes moved to `docs/archive/` (IMPLEMENTATION_COMPLETE, THEME_*, DARK_LIGHT_MODE).

### Not done
- `SVC(probability=True)` still used in the classical trainer (scikit-learn 1.9 deprecation; removal in 1.11).
- The written report has not been corrected yet (docs/REPORT_CORRECTIONS.md lists what to change).

## 2.0.0
Earlier work: ScriptInsight Flask app, handwriting validator, legacy ResNet50 loader, classical and ResNet18
training pipelines, HiEnWrite regression pipeline, Dashboard / Evaluation / History tabs.
