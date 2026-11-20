# Dataset setup

ScriptInsight never ships or invents data. You supply a labelled dataset; the pipeline splits,
trains and evaluates on it.

## Supported layouts

### A. Five independent trait labels (what the report describes): `multilabel`

```text
data/<name>/
  labels.csv
  images/...
```

`labels.csv` columns:

| column | meaning |
|---|---|
| `image` | path relative to the dataset folder |
| `participant_id` | writer / subject id (strongly recommended: keeps one writer's samples in one split) |
| `Openness`, `Conscientiousness`, `Extraversion`, `Agreeableness`, `Neuroticism` | `1` = high pole, `0` = low pole |

The labels must come from a validated instrument (e.g. BFI, TIPI) completed by the writer.
Models trained on this layout output **five independent probabilities** (one sigmoid per trait for
ResNet18; one binary classifier per trait for SVM and Random Forest). They are evaluated per trait, which
makes the report's 85% Openness / Conscientiousness target measurable.

### B. One category per image: `multiclass`

```text
data/<name>/Openness/*.jpg
data/<name>/Conscientiousness/*.jpg
...
```

This is the layout used by the legacy notebook. It only supports "which one of five categories". It does
**not** support five independent trait predictions, and the split metadata records that warning.

## Recommended public source: HWxPI (ChaLearn, ICPR 2018)

Handwritten Spanish essays by Mexican undergraduates. Each subject has a scanned page and five binary
labels (EXT, AGR, CON, STA, OPE) assigned from TIPI questionnaire norms. STA is emotional stability, so
Neuroticism = 1 − STA.

1. Register on the CodaLab competition linked from
   <https://chalearnlap.cvc.uab.cat/dataset/29/description/> and accept its terms. The data is not
   redistributable from this repository.
2. Convert each labelled split, e.g. the development data:
   ```powershell
   .\.venv313tf\Scripts\python.exe -m src.data.hwxpi `
       --data-file C:\Users\Asus\Desktop\pd\hwxpi_raw\train.data `
       --solution-file C:\Users\Asus\Desktop\pd\hwxpi_raw\train.solution `
       --image-root C:\Users\Asus\Desktop\pd\hwxpi_raw --out data\hwxpi
   ```
   (Paths are examples: use wherever you saved the files. `data\` is git-ignored.)
   Only splits with `.solution` files have labels. The challenge's final test set was unlabelled.
3. Split and train (see the README): `python -m src.data.dataset --data-root data/hwxpi`, and so on.

Limitations to state in any report: about 400 labelled subjects, one language and population, binary
labels derived from a 10-item questionnaire, and near-chance published results (best challenge AUC ≈ 0.53).

## Splitting rules (implemented in `src/data/dataset.py`)

- 80:10:10 train/validation/test over **groups**, not images:
  - group = `participant_id` when present;
  - otherwise `--group-regex` (original-sample id taken from the filename);
  - otherwise near-duplicate clusters (16×16 dHash, ≤ 6 differing bits).
- Stratified on the label (or the 5-bit label combination) when every stratum has at least 10 groups;
  otherwise grouped but unstratified, with a warning in `split_meta.json`.
- Augmentation happens only during ResNet18 training, after the split, on the training split.
- `manifest.csv` records each image's split; `split_meta.json` records the seed, grouping method, class
  distribution per split, dataset fingerprint, and a check that no group appears in two splits.
- Pre-augmented datasets (like the legacy "augmented train" folder) are risky: if copies cannot be
  grouped by id or filename, near-duplicate hashing is only a best effort for rotated or zoomed copies.
