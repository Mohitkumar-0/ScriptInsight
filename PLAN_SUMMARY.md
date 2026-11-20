# ScriptInsight completion plan (text summary)

Full plan: ScriptInsight_Completion_Plan.docx (copy it into this folder).
Plan date: 16 November 2026. Based on static review only; nothing has been executed yet.

## Goal
Run on Windows, every README feature working, every claim matching measured evidence.
Handwriting-based personality prediction is near chance (HiEnWrite regressors did not beat the
train-mean baseline; published HWxPI AUC ~0.53). The 85% target is not promised.

## Decisions (default in brackets)
- D1 Can HWxPI be obtained? [try to register; Phase 4 is independent]
- D2 Python version [3.10, one environment]
- D3 Retrain HiEnWrite models in new env [yes, same seed 42, fingerprint efe899f06db72c1d]
- D4 HiEnWrite dataset path [C:\Users\Asus\Desktop\pd\HiEnWrite-Dataset-main via SCRIPTINSIGHT_HIENWRITE_ROOT]
- D5 Hide untrained models in UI [no, show as "not trained"]
- D6 Report in scope [yes, place report .docx in folder; corrected copy only]
- D7 Accuracy experiments [no, optional]

## Remove
.venv310, .venv313, venv (macOS), all ._* files, .DS_Store, __pycache__, .pytest_cache;
stale root docs (IMPLEMENTATION_COMPLETE, THEME_*, DARK_LIGHT_MODE) -> docs/archive;
duplicate legacy .h5 if identical.

## Change / update / add
- Code: encoding="utf-8" on every read_text/open/write_text (load_json first); workers=0 default; Mac paths.
- requirements.txt: add waitress, tidy comments.
- Docs: README, QUICKSTART, docs/HIENWRITE.md for Windows + new metrics; CHANGELOG.md; version bump.
- Add: setup.bat, run.bat, tests (UTF-8 JSON, model-selection errors), PROGRESS.md.

## Train
- HiEnWrite SVR, RF, ResNet18 regression in the 3.10 env.
- If HWxPI available: SVM, RF, ResNet18 classifiers, then evaluate once on test split.
- Do not retrain legacy ResNet50.

## Phases
0 Backup (zip + weights), git branch, delete ._* and mac venvs, archive stale docs.
1 py -3.10 venv, pip install -r requirements-dev.txt, smoke imports, setup.bat/run.bat.
2 Baseline: pytest, run app, /api/health, /api/predict with samples, browser click-through, write PROGRESS.md.
3 HiEnWrite: validate, split seed 42, train_regression, train_resnet18_regression (--freeze-backbone --lr 1e-3 --epochs 60 --device cpu --workers 0), evaluate_regression, refresh doc numbers.
4 HWxPI (Path A): convert, split, train_classical, train_resnet18, evaluate. Path B: ship as untrained, documented.
5 Review/fix: UTF-8, Windows safety of scripts, image_processor/validator, frontend (script.js, dashboard.js, common.js, index.html), storage behaviour, legacy loader sanity, waitress serving.
6 Feature checklist (upload validation, drag/drop, handwriting check, predict, model selector, regression labelling, classifiers, comparison, report export, history, dashboard, evaluation, theme, API, responsive, speed).
7 Tests: new tests, optional E2E (scripts/e2e_ui.cjs), clean-venv run.
8 Docs rewrite for Windows.
9 Report: apply docs/REPORT_CORRECTIONS.md to a copy.
10 Final acceptance, tag v2.1.0, re-backup.

## Division of labour
I can read/edit text files here. I cannot run commands on your PC: you run installs, training and
pytest in PowerShell and paste the output. HWxPI registration must be done by you.

## To start
"Execute Phases 0 to 2 and report back."
