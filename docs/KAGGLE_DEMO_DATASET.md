# Kaggle handwriting dataset: demo classifiers

Written 18 November 2026. These classifiers are **demonstration models**. Read the caveats before quoting any number.

## Source
Kaggle, "Personality Prediction using Handwriting images" (Gyanendra Chaubey), licence **CC BY-NC-ND 4.0**
(non-commercial, no redistribution of derivatives). The data is **not** included in this repository.
Local location used here: `C:\Users\Asus\Desktop\pd\kaggle_handwriting\dataset` with `training_set\<Trait>` and
`test_set\<Trait>` folders (Agreeableness, Conscientiousness, Extraversion, Neuroticism, Openness).

## What it contains
- 221 original phone photos of handwriting. Every writer copied the same short passage.
- 4,076 image files after augmentation (skewed, re-lit and cropped copies of each photo). This equals the legacy
  notebook's 3,416 training + 660 test images, so this is the legacy model's dataset.
- One category per image (not five independent trait levels).
- Original photos per category: Agreeableness 38, Conscientiousness 35, Extraversion 10, Neuroticism 44, Openness 94.
  Files per original differ by category (Openness 8, Extraversion about 69, others about 22), which makes the file
  counts per category roughly equal.
- File names contain personal names. Do not show them in figures or reports.

## Problems that limit what the results can mean
1. **Labels have no documented source.** They are not questionnaire results.
2. **Each category is one block of capture times** (`IMG_20200215_HHMMSS`). A model can learn the photo session
   (lighting, paper, pen) instead of the handwriting.
3. **The same writer appears in several photos and in both English and Hindi**, and the supplied train/test split
   puts the same writers on both sides. Writer identity is not recorded, so this overlap could not be removed.
4. **Only about 22 independent photos in the test split** (about 1 Extraversion and 9 Openness photos). One photo
   moves accuracy by about 4.5 points.

## How it was split
The supplied `test_set` is ignored. All 4,076 files are merged and split 80/10/10 so that every variant of one
original photo stays in one split (group = file name before the first `-digit`):

| Split | Files | Agreeableness | Conscientiousness | Extraversion | Neuroticism | Openness |
|---|---|---|---|---|---|---|
| Train | 3,260 | 697 | 619 | 544 | 792 | 608 |
| Validation | 408 | 80 | 80 | 72 | 104 | 72 |
| Test | 408 | 80 | 80 | 72 | 104 | 72 |

## Commands (Windows PowerShell, project folder)
```powershell
$PY = ".\.venv313tf\Scripts\python.exe"
& $PY -m src.data.dataset --data-root C:\Users\Asus\Desktop\pd\kaggle_handwriting\dataset --group-regex '^(.+?)-\d' --note "DEMO ONLY: these labels have no documented source, each category was photographed in one session, and the same writers appear in several categories. Accuracy here reflects photo session and writer, not personality."
& $PY -m src.classical.train_classical
& $PY -m src.deep_learning.train_resnet18 --epochs 10 --freeze-backbone --device cuda    # --device cpu without a GPU
& $PY -m src.evaluation.evaluate
```

## Results (single run, test split evaluated once)

| Model | Validation macro-F1 | Test accuracy | Test macro-F1 |
|---|---|---|---|
| SVM (HOG + LBP) | 0.317 | 0.328 | 0.317 |
| Random Forest (HOG + LBP) | 0.316 | 0.378 | 0.386 |
| ResNet18 (frozen backbone, 10 epochs) | 0.307 | 0.272 | 0.257 |
| Majority-class baseline (always Neuroticism) | - | 0.255 | 0.081 |
| Uniform random guessing (reference) | - | 0.200 | about 0.20 |

ResNet18's validation loss was still falling after 10 epochs (1.528), so it is under-trained. The differences between
the three models are about two to three photos and are within noise. The legacy ResNet50's recorded 30.3% comes from a
different split and cannot be compared with these numbers (the legacy model saw images that are now in this test split).

## Retraining later
When better data arrives (for example a labelled set with writer IDs and questionnaire scores), place it in a
folder, re-run the four commands above with the new path (add `--group-regex` or a `labels.csv` with participant
ids as described in [DATASET.md](DATASET.md)), and the models, split and evaluation are replaced. Back up
`models\classical`, `models\deep_learning`, `artifacts\splits` and `artifacts\evaluation` first.
