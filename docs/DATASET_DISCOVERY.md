# Dataset discovery report (23 Oct 2026)

> **Update (18 Nov 2026).** The Kaggle set "Personality Prediction using Handwriting images" was found and used
> to train the SVM / Random Forest / ResNet18 classifiers as **demo models only**. It is the legacy model's own
> data (4,076 augmented images from 221 photos), has no documented label source, one photo session per category
> and overlapping writers. See [KAGGLE_DEMO_DATASET.md](KAGGLE_DEMO_DATASET.md). The conclusion below (no
> *validated* labelled dataset was found) still stands. HWxPI could not be obtained.

> **Update (23 Oct 2026, later).** The project owner downloaded **HiEnWrite** (Hindi-English handwriting with
> five numerical Big Five scores; CC BY-NC 4.0). It is now used for *experimental five-trait regression*;
> see [HIENWRITE.md](HIENWRITE.md). It does not provide binary or categorical labels, so the conclusion below
> still holds for the SVM / Random Forest / ResNet18 **classifiers**.

**Result: no usable labelled handwriting dataset was found.** SVM, Random Forest and ResNet18 training
are therefore blocked. No labels were invented and no substitute dataset was used.

## Where we looked

- The project folder, its parent (`~/Desktop`), `~/Downloads` and `~/Documents`, to depth 4 to 7. We
  searched for folders named after the traits or `archive` / `augmented*`, for archives
  (`.zip`, `.tar*`, `.7z`, `.rar`), and for label-like files (`*big*five*`, `*bfi*`, `*personality*.csv`,
  `*label*.csv`, `*trait*.csv`).
- The training notebook's outputs, for the dataset path and counts.
- The web: the dataset's public origin, and public handwriting / Big Five datasets.

## Candidates

| Candidate | Contents | Labels / provenance | Usable? |
|---|---|---|---|
| Notebook dataset: local Windows folder `archive/` with `augmented train` (3,416 images) and `augmented test` (660), one folder per trait, 1024×1024 grayscale | **Not present on this machine.** Only the four sample JPGs (`1.1, 1.2, 3.1, 3.3.jpg`) are in the repo, and they have no labels | One category per image. Source and labelling method undocumented. `archive` is Kaggle's default download name, but no matching public dataset was identified. Train counts are all multiples of 8 (pre-augmented) | **No** (missing; single-label; unverified provenance) |
| `~/Downloads/handwriting_personality_large_dataset.csv` (2,000 rows) | Numeric columns `Openness … Neuroticism`, `Feature_1..15`, `Writing_Speed_wpm`, `Age`, referring to `sample_N.jpg` | Every value is uniform on [0, 1] (SD ≈ 0.289 = 1/√12), trait–feature correlations ≈ 0, and none of the referenced images exist: **synthetic random data** | **No** |
| `~/Downloads/HWxPI Handwritten texts for Personality Identification results.csv` | 9-row leaderboard (`User, AUC performance`) | Competition results only (best AUC 0.5314, i.e. near chance) | **No** (not data) |
| `~/Downloads/handwriting_personality_ai-main.zip` | Another project (Streamlit, rule engine) | Its README says "No training dataset required" | **No** |
| `~/Desktop/scriptInsight/` | Lovable/React UI prototype | No data | **No** |
| **HWxPI** (ChaLearn, ICPR 2018), public | Scans of handwritten essays plus transcripts, subject ids | Five binary Big Five labels per subject from TIPI norms; columns EXT AGR CON STA OPE | **Suitable**, but **access requires CodaLab registration and accepting its terms**, which the project owner must do |

## Upload-bug evidence found during discovery

The screen recording `~/Desktop/Screen Recording 2026-10-22 at 11.41.04 PM.mov` (18.9 s) shows:
- The browser is **Safari**. After picking `1.1.jpg` and pressing *Upload*, the page still shows
  "Choose a handwriting image" (no filename, no preview).
- Clicking *Analyze* shows "Please choose an image file first." as a native **`alert()`** dialog.
- The `script.js` on disk from 22:48 that day never called `alert()`, so Safari ran an older or different
  copy of the script: either a cached file (the app then sent no cache headers) or a server started from an
  earlier copy of the project. Static assets now carry file-version query strings and `Cache-Control:
  no-cache`, and the current script passes the same flows in WebKit and Chrome (see test results).

## To unblock training

1. Obtain HWxPI (see `docs/DATASET.md`) or another dataset with writer-level Big Five questionnaire labels
   and a licence that allows academic use.
2. Convert it to `labels.csv` format, then run the split, train and evaluate commands in the README.
