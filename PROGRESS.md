# PROGRESS

Updated 17 November 2026. Plan: ScriptInsight_Completion_Plan.docx / PLAN_SUMMARY.md.

## Result of Phases 0 to 2

| Phase | Status |
|---|---|
| 0 Backup and cleanup | DONE. Backup: C:\Users\Asus\Documents\HandParsona_backup_20261117_0438 (weights verified). ._* files and macOS venvs (.venv310, venv) removed. Stale docs in docs/archive. |
| 1 Environment | DONE using the existing Windows env `.venv313` (Python 3.13.3). Python 3.10 is not installed and is not required for this. |
| 2 Baseline | DONE. See below. |
| 3 HiEnWrite retrain | NOT NEEDED. The saved models load and run in `.venv313`. (They were trained with torch 2.14.1 / scikit-learn 1.9.1, the same versions.) |
| 4 Classifiers | WAITING for HWxPI access (decision D1). |

## Phase 2 baseline (Python 3.13.3, Windows 11)

- **Tests:** 121 passed, 4 skipped, 0 failed (98 s). Reasons for the 4 skips not yet captured (`pytest -rs`).
- **App:** starts in under a minute. `/` 200, `/api/health` 200 status ok, `/api/info` 200, `/api/metrics` 200.
- **Models loaded:** hienwrite_resnet18, hienwrite_svr, hienwrite_rf.
- **Models unavailable:** legacy_resnet50 (TensorFlow not installed in this env), resnet18 / svm / random_forest (never trained, as expected).
- **Predictions:** /api/predict returned 200 for all four sample images, 50 to 110 ms per model.
- **Metrics:** HiEnWrite evaluation available; classifier evaluation (artifacts/evaluation) not available, legacy recorded metrics available.
- **Primary model** is hienwrite_rf (lowest validation normalised RMSE) because no classifier is loaded.

### Observations
1. The SVR returns almost the same scores for every image (Agreeableness 27.10, 27.12, 26.99, 27.09 for the four samples), i.e. it predicts about the training mean. This matches the measured test result (no better than the mean baseline). The ResNet18 varies more between images (Neuroticism 15.1 to 26.6) but its validation correlation is about 0.1, so the variation is noise, not signal. The UI warnings must stay prominent.
2. The environment has two supported stacks: Python 3.10 (requirements.txt, TensorFlow 2.13, legacy model) and Python 3.13 (requirements-py313.txt, no TensorFlow). requirements-py313.txt was added from the installed packages.
3. Warnings to fix in Phase 5: scikit-learn 1.9 FutureWarning that `SVC(probability=True)` is deprecated (removal in 1.11, affects train_classical); `float(loss)` on a tensor with grad in src/deep_learning/train_resnet18.py line 98 (use `loss.item()`); Flask development server notice (use waitress for anything beyond local use).
4. Mac paths remain in artifacts/hienwrite/splits/split_meta.json (data_root). Only matters if retraining or re-evaluating (use `--data-root`).
5. HiEnWrite dataset not on this PC (not needed unless retraining).

## Legacy model (17 Nov 2026, after Phase 2)
DONE. `.venv313tf` (Python 3.13.3 + TensorFlow 2.21.0 + Keras 3.15.1) loads legacy_resnet50 and answers /api/predict with 200. All four usable models now load: legacy_resnet50, hienwrite_resnet18, hienwrite_svr, hienwrite_rf. `run.bat` uses this environment first. Python 3.10 is no longer needed. Note: with the legacy model loaded it becomes the default (primary) model.

## Files added this session
setup.bat, run.bat (uses .venv, .venv313tf or .venv313), requirements-py313.txt, scripts\phase0_cleanup.ps1, scripts\phase2_baseline.ps1 (optional `-Python <path>`), scripts\add_legacy_tensorflow.ps1.

## Phase 5 fixes applied (17 Nov 2026) - TESTED: 128 passed, 0 skipped, 0 failed (88 s, .venv313tf)
- src/services/prediction_service.py: load_json reads UTF-8 explicitly.
- src/data/dataset.py, src/data/hienwrite.py, src/data/hwxpi.py, scripts/extract_legacy_records.py: every CSV/JSON/Markdown read and write now specifies UTF-8 (Windows default is cp1252).
- src/deep_learning/train_resnet18.py: `float(loss.detach())` removes the PyTorch warning.
- waitress added to requirements.txt and requirements-py313.txt; serve.bat starts the production server.
- tests/test_load_json.py added (3 tests).
- NOT changed on purpose: `SVC(probability=True)` in the classical trainer (scikit-learn 1.9 deprecation, removal in 1.11). Replacing it changes how the SVM scores are calibrated, so it needs a decision; the current pinned stacks still work.
- Decision D1 (HWxPI): postponed by the user. Classifier slots stay "not trained" for now.

## Phase 6 and 8 (17 Nov 2026)
- Phase 6 browser checklist: confirmed working by the user (app, history, dark mode, report download, dashboard, evaluation tab). Untrained classifiers show "Train it first", as designed.
- Phase 8 docs DONE: README.md, QUICKSTART.md, docs/HIENWRITE.md, docs/DATASET.md rewritten for Windows; CHANGELOG.md added. APP_VERSION left at 2.0.0 (a bump could break tests that check it; not checked).
- Not yet reviewed for stale text: docs/DATASET_DISCOVERY.md, docs/HANDWRITING_VALIDATION.md, docs/REPORT_CORRECTIONS.md.

## Kaggle handwriting dataset inspected (17 Nov 2026)
Location: C:\Users\Asus\Desktop\pd\kaggle_handwriting\dataset (training_set / test_set, one folder per trait; images not opened, only the file listing was read).

| Trait | Train | Test | Total |
|---|---|---|---|
| Agreeableness | 30 | 8 | 38 |
| Conscientiousness | 27 | 8 | 35 |
| Extraversion | 8 | 2 | 10 |
| Neuroticism | 36 | 8 | 44 |
| Openness | 76 | 18 | 94 |
| **All** | **177** | **44** | **221** |

Findings from the file names (not yet confirmed by looking at the images):
1. Only 221 photos, very imbalanced (Extraversion 10, Openness 94; majority-class baseline about 42.5%).
2. Photos are phone images named IMG_20200215_HHMMSS. Each trait folder holds one contiguous block of capture times (Agreeableness 16xxxx, Neuroticism 17xxxx, Conscientiousness 174xxx-180xxx, Openness 181xxx-193xxx, Extraversion 200xxx). A classifier can learn the photo session instead of the handwriting.
3. The same writer appears in both splits (several file names appear in both training and test, in English and Hindi versions, under the same category; names omitted here). The same writer also appears in English and Hindi. Test results would be inflated by writer leakage.
4. File names contain personal names: do not publish them in figures or the report.
5. Likely (unverified) source of the legacy model's data: the notebook reports 3,416 augmented training and 660 augmented test images, about 19x and 15x these 177 and 44 originals.
6. Licence from the Kaggle page: CC BY-NC-ND 4.0 (non-commercial, no redistribution of derivatives).

Conclusion: a model trained on this cannot show anything about personality; any accuracy would reflect session and writer identity.

UPDATE (18 Nov 2026): the folders now contain augmented copies (training Openness 608 files = 76 originals x 8; test Extraversion 112 files = 2 originals x 56). Still 221 original photos. All writers copy the same sentence. Variants of one page are skewed/re-lit copies (checked by viewing 1-1.jpg and 1-5.jpg).

DECISION (user): Option B, train as a labelled DEMO only; retrain later if more/better data arrives. Code added: nested-layout loading in src/data/dataset.py (merges training_set + test_set, new split), `--note` warning option, majority-class baseline in src/evaluation/evaluate.py, tests/test_nested_layout.py. Split rule: group by original photo with `--group-regex '^(.+?)-\d'`. Residual leakage: same writer across photos is NOT removed (no writer ids).

SPLIT RESULT (18 Nov 2026): tests test_nested_layout + test_training_pipeline pass (2). Split made: 4,076 images, 221 groups (original photos), train 3,260 / val 408 / test 408, no group in two splits. 4,076 equals the legacy notebook's 3,416 + 660 exactly, so this IS the legacy model's dataset (augmented). Per-class files (train/val/test): Agreeableness 697/80/80, Conscientiousness 619/80/80, Extraversion 544/72/72, Neuroticism 792/104/104, Openness 608/72/72. Files per original differ by class (Openness 8, Agreeableness/Conscientiousness/Neuroticism about 22, Extraversion about 69), which balanced the file counts: majority-class baseline is about 25% (Neuroticism), not 42%. The test split holds only about 22 independent photos (Extraversion about 1, Openness 9): one photo is about 4.5 percentage points. The legacy model cannot be fairly evaluated on this split (it trained on images that are now in test).

TRAINING RESULT (18 Nov 2026): GPU enabled (torch 2.14.1+cu126, RTX 3050 Laptop 4.3 GB, driver 610.47). Validation macro-F1: SVM 0.3171, Random Forest 0.3155, ResNet18 0.3073 (10 epochs, frozen backbone, val loss still falling at 1.528). Test (evaluated once): SVM acc 0.3284 / F1 0.3168; Random Forest acc 0.3775 / F1 0.3855; ResNet18 acc 0.2721 / F1 0.2574; majority baseline acc 0.255 / F1 0.081; chance 0.20. Differences between models are within noise (about 22 test photos). Written up in docs/KAGGLE_DEMO_DATASET.md; README status table updated. STILL TO CHECK by the user: app Dashboard/Evaluation/prediction with the new models; full pytest after the torch swap (expect 129 passed).

APP CHECK + TESTS (18 Nov 2026): pytest 126 passed, 3 failed. The 3 failures were tests that assumed the legacy model is the default; the SVM is now the default (best validation macro-F1, 0.3171 vs 0.3155 RF vs 0.3073 ResNet18, so the margin is tiny). Fixed by giving those tests a `legacy_client` fixture that points every other model at a missing file, plus a new test that uses the real on-disk models. NOT YET RE-RUN. User confirmed: slots no longer say "Train it first", default model is SVM. The Evaluation tab showed the classifier table but not the baseline, the DEMO warnings or the confusion matrices, and still said the legacy dataset is unavailable. Fixed in static/js/dashboard.js (baseline row, warnings callout, copies note, confusion-matrix disclosure, legacy wording). NOT YET RE-CHECKED in the browser.

FINAL CHECK (18 Nov 2026): pytest 130 passed. Evaluation tab confirmed by the user: baseline row, DEMO warnings, copies note and confusion matrices all show; legacy wording corrected. Default model is the SVM (best validation macro-F1).

STATUS: all planned work is done except (a) the written report (file not provided; docs/REPORT_CORRECTIONS.md), (b) optional longer ResNet18 training, (c) the SVC(probability=True) deprecation fix, (d) HWxPI or other validated data if it ever becomes available, (e) a fresh backup of the new weights (models/classical, models/deep_learning, artifacts/splits, artifacts/evaluation) and a git commit.

## Next
- Optional: `scripts\add_legacy_tensorflow.ps1` to try the legacy model on Python 3.13 (separate env, nothing existing is changed).
- Phase 5 fixes, Phase 6 browser checklist, Phase 8 docs.
- Phase 4 when HWxPI is available.
