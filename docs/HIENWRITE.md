# HiEnWrite: five-trait regression (experimental)

ScriptInsight can train and serve **experimental** models that predict the five Big Five scores of the
HiEnWrite dataset from a handwriting image. This page covers the dataset, the exact data preparation,
the commands, where the artifacts are saved, and the **measured results**.

> **Read the results section first.** On the held-out test split, none of the trained models beats a
> constant prediction of the training-set mean. The models are integrated and working, but they have no
> demonstrated ability to predict these labels. Handwriting is not a validated way to assess personality.

## Dataset and licence

**HiEnWrite: A Hindi-English Bilingual Dataset for Big Five Personality Detection.** Saksham Checker,
Madhuri Yadav and Rahul Katarya. *ACM Transactions on Asian and Low-Resource Language Information
Processing* (2025), doi:[10.1145/3756010](https://doi.org/10.1145/3756010). Data version 1.0.0,
<https://github.com/sakshamchecker/HiEnWrite-Dataset>.

```bibtex
@article{checker2025hienwrite,
  title={HiEnWrite: A Hindi-English Bilingual Dataset for Big Five Personality Detection},
  author={Checker, Saksham and Yadav, Madhuri and Katarya, Rahul},
  journal={ACM Transactions on Asian and Low-Resource Language Information Processing},
  year={2025},
  doi={10.1145/3756010}
}
```

**Licence: CC BY-NC 4.0.** You must credit the authors, link the licence, and indicate changes. **No
commercial use.** That covers the models trained on it and their predictions too. The dataset is not
included in this repository. Download it yourself, by default to `%USERPROFILE%\Downloads\HiEnWrite-Dataset-main`;
for another location, set `SCRIPTINSIGHT_HIENWRITE_ROOT` or pass `--data-root`.

**Changes made for this project** (licence attribution): no files were modified. The pipeline excludes
some rows and treats one label as missing, as described below. Exclusions are listed in
`artifacts/hienwrite/validation/validation_report.md`.

### Structure

```text
HiEnWrite-Dataset-main/
  data/merged_data.csv   690 rows: Unnamed: 0, image, Extroversion, Agreeableness, Conscientiousness,
                         Neuroticism, Openness to Experience, Age, Gender
  images/                660 JPEG files, named <number><e|h>.jpeg
  readme.md, LICENCE, citation.cff
```

- `e` images are the same English passage copied by hand, and `h` images are a Hindi essay. All are phone
  photos of varying size (48 to 5759 px wide), lighting and orientation.
- The five scores are integers. Ranges over all 690 rows: Extroversion 0–40, Agreeableness 11–40,
  Conscientiousness 7–40, Neuroticism 2–37, Openness 8–40. The readme says they are self-reported
  questionnaire scores; it does not name the questionnaire.
- Internal target names → CSV columns: `Extraversion → Extroversion`, `Agreeableness`,
  `Conscientiousness`, `Neuroticism`, `Openness → Openness to Experience`
  (`src/data/hienwrite.py: TARGET_COLUMNS`).

## Data validation (findings on the real files)

`python -m src.data.hienwrite validate` re-checks everything and writes the report. These are the counts
it measured:

| | Rows |
|---|---|
| CSV rows (no missing values, no duplicate rows or image ids) | 690 |
| Matched to an image file | 660 |
| Missing image files (12 participants have neither image, 6 have one; no file on disk is absent from the CSV) | 30 |
| Unreadable: 0-byte files (`119e, 123h, 134h, 151e, 291e, 348e, 60e`) | 7 |
| Unreadable: truncated JPEG (`52h`) | 1 |
| Too small (< 128 px): `176h` is a 48×48 portrait photo, not handwriting | 1 |
| Duplicate photographs, labels conflict → **all copies excluded** | 68 |
| Duplicate photographs, labels agree → later copies excluded | 5 |
| **Usable samples** | **578** (298 English, 280 Hindi) |
| Inferred participants / split groups | 303 / 295 |

**Openness is constant in Hindi rows.** All 345 Hindi rows have `Openness to Experience = 24`, a standard
deviation of 0. English rows range 8–40 (sd 5.5). The other four traits are identical between the e and h
rows of every participant. The Hindi Openness value therefore carries no information. It looks like a
placeholder, and training on it would teach "Hindi script → 24". It is **masked** (treated as missing) for
Hindi rows. Openness is trained and evaluated on English images only. The value is not replaced or
imputed. Detection is data-driven: any target that is constant within a language subset of at least
20 rows is masked.

**Duplicate photographs.** Byte-identical files, and the same photograph re-encoded or colour-shifted
(16×16 difference hash within 6 of 256 bits; all 40 such pairs were checked visually), appear under
different participant numbers. For example, `231e, 232e, 236e, 239e, 248e, 250e, 283e, 297e, 297h, 310e`
are one photo with ten different label sets. The correct label cannot be known, so such clusters are
excluded entirely. Where the copies' labels agree (`144e/144h`, `337e/337h`, `146/172`, `157h/295h`),
only the first row is kept.

**Writer identity.** The CSV and readme have **no writer-id column**. These facts point to the numeric
prefix of the image id identifying a participant:

1. For all 345 numbers, the `e` and `h` rows have identical Extroversion, Agreeableness, Conscientiousness,
   Neuroticism, Age and Gender.
2. The paper reports handwriting "from over 400 authors" and "approximately 800 images", i.e. about two per
   author. Ids run from 0 to 403.

The number is therefore used as an **inferred** participant id, not a documented one. Participants that
share a duplicate photograph are merged into one group. Limitation: if one person contributed under two
numbers, they could appear in two splits.

## Split procedure

`python -m src.data.hienwrite split`:

- Whole **groups** are assigned to train / validation / test at **70 / 15 / 15**, using a NumPy permutation
  with **seed 42**. Both languages of a participant always land in the same split.
- Checks, which fail loudly: no group, participant or image hash appears in two splits.
- Result: **405 / 87 / 86 images** (207 / 44 / 44 groups). The test split has 86 images; Openness is valid
  for 46 of them.
- Dataset fingerprint `efe899f06db72c1d`; CSV sha256 recorded in `split_meta.json`.
- Everything fitted is fitted on the training split only: feature scaler, PCA, target standardisation,
  ResNet target mean/std. Augmentation is applied only to training images. Hyper-parameters, checkpoints
  and the choice between ResNet configurations use the validation split. The test split is read only by
  `evaluate_regression`, which was run after all models were fixed.

## Models

All three models predict the five raw scores with one output per trait. Openness is learned from English
rows only. The classifiers (multi-class, binary), their softmax outputs and the HWxPI label conversion are
not used.

| Model | Input | Estimator | Selection (validation) |
|---|---|---|---|
| `hienwrite_svr` | HOG (6084) from a 224×224 aspect-preserving image | per trait: StandardScaler → optional PCA → RBF SVR, standardised target | grid PCA {64, none} × C {0.3, 1, 3} × ε {0.1, 0.5}, per-trait RMSE |
| `hienwrite_rf` | HOG + LBP (6262) | per trait: RandomForestRegressor, 500 trees | max_features {sqrt, 0.1} × min_samples_leaf {5, 20}, per-trait RMSE |
| `hienwrite_resnet18` | grayscale ×3, 224×224 aspect-preserving, ImageNet normalisation | torchvision ResNet18 (ImageNet weights), `fc` replaced by Linear(512→5); masked MSE on standardised targets | early stopping on validation loss (patience 8) |

Preprocessing is shared by training and inference through `prepare_gray()`. It applies a 3×3 Gaussian
denoise, then fits the photo inside a 224×224 square, padding with the photo's median gray, so slant and
letter proportions are not distorted. Each artifact stores its preprocessing parameters.

ResNet18 augmentation (training only): RandomAffine (±10°, 3% translation, 0.95–1.05 scale, 5° shear)
and brightness/contrast jitter 0.2. No flips. Three configurations were compared **on validation loss
only** (`artifacts/hienwrite/resnet18_candidates.json`):

| Run | Best epoch / run | Val loss | Val macro RMSE |
|---|---|---|---|
| full fine-tune, lr 1e-4, 224 px | 2 / 10 | 0.917 | 5.67 |
| **frozen backbone (linear probe), lr 1e-3, 224 px** (selected) | 12 / 20 | **0.814** | 5.39 |
| full fine-tune, lr 3e-5, 384 px | 1 / 9 | 0.821 | 5.39 |

Full fine-tuning overfitted within one or two epochs. The selected run was retrained with the documented
command into `models/hienwrite/` and reproduced the same validation loss (0.81434, epoch 12). Device: Apple
M4 GPU (MPS), used after its outputs matched CPU on a check batch. Training took about 30 s.

## Results (held-out test split, measured 2026-10-22)

The test split has 86 images from 44 participant groups. Openness n = 46, English only. Macro = unweighted
mean over the five traits. The baseline predicts the training-split mean of each trait. The last column
is the participant-group bootstrap (2,000 resamples) 95% interval of the model's macro RMSE minus the
baseline's: **positive means worse than the baseline.**

| Model | MAE | RMSE | R² | Pearson r | RMSE − baseline, 95% CI |
|---|---|---|---|---|---|
| Train-mean baseline | 4.322 | 5.244 | −0.025 | undefined (constant) | — |
| ResNet18 regression | 4.613 | 5.869 | −0.286 | −0.044 | [+0.220, +1.012] |
| SVR (HOG) | 4.402 | 5.404 | −0.091 | −0.052 | [+0.014, +0.306] |
| Random Forest (HOG + LBP) | 4.437 | 5.407 | −0.088 | −0.080 | [+0.042, +0.277] |

Per-trait test RMSE / Pearson r:

| Trait (n) | Baseline RMSE | ResNet18 | SVR | Random Forest |
|---|---|---|---|---|
| Extraversion (86) | 6.50 | 7.17 / −0.15 | 6.71 / −0.23 | 6.80 / −0.18 |
| Agreeableness (86) | 4.61 | 4.41 / +0.29 | 4.65 / −0.03 | 4.67 / −0.11 |
| Conscientiousness (86) | 5.65 | 6.90 / −0.29 | 5.71 / −0.03 | 5.80 / −0.04 |
| Neuroticism (86) | 4.76 | 5.35 / +0.08 | 4.84 / +0.02 | 4.82 / −0.00 |
| Openness (46, English) | 4.70 | 5.51 / −0.16 | 5.11 / +0.00 | 4.95 / −0.07 |

**Interpretation.**
- Every model has a higher macro RMSE than the constant baseline, and every 95% interval lies entirely
  above zero. In ≥ 98% of bootstrap resamples, each model was worse than the baseline.
- R² is negative throughout, and the macro correlations are about zero.
- ResNet18's Agreeableness correlation (+0.29, p = 0.006) comes with a Conscientiousness correlation of
  −0.29 (p = 0.008) in the same model. These are 15 uncorrected tests on 86 images, so isolated nominal
  significance is expected by chance. Neither is evidence of a real effect.
- Validation results were similar: macro r of 0.01 for SVR, 0.05 for RF and 0.10 for ResNet18.
- The paper reports a best correlation of 0.411 (VGG19). This run does not reproduce anything close to it.
  The splits differ: this one is participant-grouped, de-duplicated, and masks the constant Hindi
  Openness. The paper's protocol has not been checked here, so the difference is not explained.

Artifacts are in `artifacts/hienwrite/evaluation/`:
- `latest.json`: all metrics, per-language breakdown, bootstrap intervals and configs.
- `predictions_<model>.csv`: per-image true and predicted scores, plus the validity mask.
- Plots: `test_rmse_by_trait.png`, `scatter_<model>.png` and `resnet18_training_curves.png`.

**Not comparable with the legacy ResNet50.** That model is a single-label, 5-category classifier trained on
an undocumented dataset with a recorded 30.3% accuracy. These HiEnWrite models predict five numerical
scores, and their metrics are RMSE and correlation. HiEnWrite results are not HWxPI results either; no
HWxPI model has been trained.

## Commands

Use the project interpreter (Windows PowerShell, from the project folder). The results above were produced with
these steps on a Mac; on Windows only the interpreter and the path change. The manifests under
`artifacts/hienwrite/` store the dataset path of the machine that made them, so run steps 1 and 2 with your own
path before retraining (the fingerprint `efe899f06db72c1d` must come out unchanged).

```powershell
$PY = ".\.venv313tf\Scripts\python.exe"
$DATA = "C:\Users\Asus\Desktop\pd\HiEnWrite-Dataset-main"    # wherever you downloaded it

# 1. Validate the dataset and write the data manifest + report (about 3 s)
& $PY -m src.data.hienwrite validate --data-root $DATA
# 2. Participant-grouped 70/15/15 split (seed 42)
& $PY -m src.data.hienwrite split --seed 42
# 3. Classical regressors (SVR on HOG, Random Forest on HOG+LBP; about 30 s)
& $PY -m src.classical.train_regression
# 4. ResNet18 smoke test: 32 images, 1 epoch, writes to artifacts/hienwrite/smoke/, reloads the checkpoint
& $PY -m src.deep_learning.train_resnet18_regression --smoke
# 5. ResNet18, selected configuration (about 30 s on an Apple M4 GPU; on a Windows CPU expect minutes; --device cpu forces CPU)
& $PY -m src.deep_learning.train_resnet18_regression --epochs 60 --freeze-backbone --lr 1e-3 --workers 0
#    other flags: --batch-size --image-size --loss {mse,smoothl1} --patience --workers --seed --no-pretrained
# 6. Test-split evaluation (run once, after all choices are fixed)
& $PY -m src.evaluation.evaluate_regression
# 7. Tests
& $PY -m pytest -q
# 8. App: http://127.0.0.1:5001
.\run.bat
```

Retraining overwrites the models and the evaluation under `models/hienwrite/` and `artifacts/hienwrite/`; the
numbers in this page would then need to be refreshed from the new `latest.json`. Back up first.

## Saved artifacts

| Path | Content | In git |
|---|---|---|
| `artifacts/hienwrite/validation/data_manifest.csv` | every CSV row: path, inferred participant, group, status, exclusion reason, sha256, size, targets, masks | yes |
| `artifacts/hienwrite/validation/validation_report.{json,md}` | counts, exclusions, duplicates, masked targets, distributions | yes |
| `artifacts/hienwrite/splits/split_manifest.csv`, `split_meta.json` | usable rows with split; seed, ratios, groups per split, distributions, leakage checks | yes |
| `artifacts/hienwrite/resnet18_candidates.json` | validation comparison of the three ResNet18 runs | yes |
| `artifacts/hienwrite/evaluation/` | test metrics, predictions, plots | yes |
| `models/hienwrite/svr_hog.joblib`, `random_forest_hog_lbp.joblib` (+ `.json` sidecars) | per-trait estimators, preprocessing, feature config, selected params, validation metrics, target stats | sidecars only |
| `models/hienwrite/resnet18_regression.pt` (+ `.json`), `resnet18_regression_history.json` | checkpoint with metadata (target mean/std, preprocessing, augmentation, versions), epoch history | metadata only |
| `artifacts/features/hienwrite_*.npz` | cached HOG+LBP features (regenerable) | no |

The `.joblib` files are pickles. Load only files you created yourself. They load only with the
NumPy/scikit-learn versions used to train them (scikit-learn 1.9.1; they also load with numpy 2.5.5 on
Windows / Python 3.13.3, verified 17 Nov 2026). The app reports any other version as "cannot be loaded … retrain".

## Inference

- **Web app:** the *Primary model* menu on the Analyze tab lists the loaded models. *Automatic* keeps the
  existing default: classifiers first (the legacy ResNet50 when it loads), so HiEnWrite never silently
  replaces it. Only when no classifier is available does it use the HiEnWrite model with the lowest
  validation normalised RMSE, which is currently the Random Forest. Regression results show five scores
  on the training-label range, marked **Experimental**, with a warning.
- **API:** `POST /api/predict` with `file` and, optionally, `model=hienwrite_resnet18|hienwrite_svr|hienwrite_rf`.
  The response includes `prediction.task = "regression"`, `trait_scores`, `training_label_stats` and
  `experimental: true`.
  - An unknown model name returns 400 `unknown_model`.
  - A model that is not loaded returns 503 `model_unavailable`; another model is never substituted.
  - `model_predictions` lists every model's output and status.
  - `/api/info` reports each model's state.
  - `/api/metrics` contains a `hienwrite` block, read from the files above.
- **Python:**
  ```python
  from src.deep_learning.predict_resnet18 import ResNet18Model
  from src.preprocessing.image_processor import decode_image_file
  model = ResNet18Model('models/hienwrite/resnet18_regression.pt')
  model.predict_gray(decode_image_file('1.1.jpg').gray)  # {'task': 'regression', 'scores': {...}}
  ```
- A missing checkpoint is reported as `not_trained`. Random weights are never used.
- **Observed on Windows (17 Nov 2026, four sample photos):** the SVR returns almost the same scores for every
  image (for example Agreeableness 27.10, 27.12, 26.99 and 27.09), i.e. close to the training mean, as its test
  result predicts. The ResNet18 varies more between images (Neuroticism from 15.1 to 26.6), but with a validation
  correlation of about 0.1 that variation is noise, not signal. Treat every HiEnWrite output as a demonstration.

## Limitations

- **No predictive signal was found**, and nothing here supports using the outputs for anything.
- The labels are self-reported questionnaire scores whose instrument is not documented. The dataset's
  Hindi Openness values are unusable.
- Only 578 usable images from about 300 participants remain, and the test split is small (86 images; 46
  for Openness), so the confidence intervals are wide.
- Writer ids are inferred, not documented. The duplicate-photo clusters show that some image-to-participant
  assignments in the source data are wrong. Undetected mistakes of the same kind may remain.
- 30 images are missing and 8 are unreadable.
- The English images all copy the same passage, while the Hindi images are essays. Content is not
  controlled across languages.
- Images are downscaled to 224×224, which removes fine stroke detail. Patch-based models at native
  resolution were not tried.
- One split seed was evaluated, with no repeated or nested cross-validation.
- MPS training is not bit-for-bit deterministic across hardware. The run here was reproduced on the same
  machine.
- Unverified observation: the legacy notebook's "augmented test" folder also had 660 images. Whether it was
  derived from HiEnWrite is unknown.
