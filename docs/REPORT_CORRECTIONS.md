# Report-correction checklist

For: `report final final final (2).docx` (ScriptInsight, VI Sem CSE(DS) 2025-26).
The original report has not been modified. Each item says what the report claims, what the code and
evidence show (as of 23 Oct 2026), and a suggested correction.

## A. Content errors

- [ ] **Chapter 2 preamble is from another project.** The three opening paragraphs and the sentence
  "…addressed by the proposed Smart Indoor Hospital Navigation System using QR Code with Intelligent Path
  Finding" describe indoor navigation (GPS, RFID, BLE, hospitals). *Replace with a preamble on
  handwriting-based personality prediction.*
- [ ] **FR "Classical Feature Extraction" has a heading but no text.** *Add:* "The system shall extract HOG
  (9 orientations, 16×16 px cells, 2×2-cell blocks, L2-Hys; 6,084 values) and uniform LBP (P=8, R=1 on a
  4×4 grid, plus P=16, R=2 global; 178 values) from the 224×224 grayscale image and concatenate them into a
  6,262-value vector."
- [ ] **Figure numbering.** UI-3 and UI-4 are both called "Figure 4.5"; UI-4 should be Figure 4.6.
  Section numbering is also inconsistent ("1.1.. Existing System", "2. Workflow Description",
  "4. User Interface Components", "1. Dashboard Home Screen").
- [ ] **Literature survey and reference list do not line up.** Survey item [5] is "Human Personality
  Identification … with SVM (2022)", but reference [5] is Saraswal et al. (KNN). Items [1] and [7] appear to
  describe the same adaptive DCRNN work. Check every in-text number against the reference list.
- [ ] Typos: "Specifiaction", "Convulational", "ResNet18'sinput", "AshishSaraswalet".

## B. Technology statements that conflict with the implementation

- [ ] **React.js frontend** (Ch. 1 Proposed System, Objectives, Tools, Table 1.8; Ch. 3 Software
  Requirements; Ch. 4.3 and Frontend-Backend Interaction). The project uses **Flask templates with HTML,
  CSS and vanilla JavaScript**. *Replace "React.js 18+, Node.js 18+" with that.*
- [ ] **"/predict endpoint".** Now true (`POST /predict` is an alias of `POST /api/predict`). *Mention both
  routes and the `file` multipart field.*
- [ ] **"PyTorch / TensorFlow".** Both are used: PyTorch for ResNet18, TensorFlow 2.13 for the legacy
  ResNet50 demo. *State which model uses which framework.*
- [ ] **UI-1 "navigation header provides access to model selection".** There is no model selector: every
  model runs on each upload and the primary model is the one with the best validation macro-F1. *Reword to
  "model comparison".*
- [ ] **UI-1 "home screen presents model accuracy, training loss, … epochs trained".** The Dashboard shows
  these only where they are recorded or measured, with the source labelled. *Say so.*

## C. Claims not supported by evidence

- [ ] **"Each image is associated with Big Five labels derived from validated self-report questionnaire
  scores (e.g. BFI)."** The only dataset ever used (the legacy notebook's) has one category folder per image,
  with no questionnaire data and no documented source. *Describe the dataset actually used, or HWxPI (TIPI,
  binary per trait) once it is obtained.*
- [ ] **"Multi-label SVM … predict Big Five trait probabilities"; "simultaneous classification of all five
  dimensions"; "one-vs-rest (OVR) multi-label configuration".** Implemented in code (one binary classifier
  per trait), but **untrained**. The only working model is single-label. *Keep it as design and report it as
  not yet trained.*
- [ ] **"fine-tuned ResNet18".** The architecture and training code exist and are tested, but **no trained
  ResNet18 exists**. The working model is a Keras **ResNet50** with a frozen backbone (its fine-tuning
  phase never improved and was not saved).
- [ ] **"80:10:10 stratified split".** Implemented (grouped by writer, stratified where possible) but not
  yet run on real data. The legacy model used an 80/20 random split of a pre-augmented folder plus a separate
  test folder (about 67:17:16), with likely leakage between augmented copies.
- [ ] **NFR "≥ 85% validation accuracy for Openness and Conscientiousness".** **Not met.** The legacy model
  has 45.6% best validation accuracy overall and 30.3% test accuracy (five categories, chance ≈ 20%), and it
  has no per-trait accuracy. Published HWxPI results are near chance (best AUC ≈ 0.53). *Report the
  measured values; do not imply the target was reached.*
- [ ] **NFR "inference within three seconds".** Met on the test machine (Apple M4, CPU): legacy model
  median ≈ 0.05 s end to end. Not measured on the report's minimum spec (i5, 8 GB).
- [ ] **NFR "operate across Windows, Linux and macOS".** Dependencies resolve for all three, but the app has
  only been run on macOS.
- [ ] **Problem statement and objectives framing** ("predicting Big Five traits with measurable accuracy",
  "objective … alternative to traditional personality assessment", use in recruitment). *Add that predictive
  validity is not established, that results are experimental, and that the system must not be used for
  hiring or clinical decisions.*
- [ ] **"The best-performing model is identified and used for real-time inference."** Implemented as
  "highest validation macro-F1 among trained models", but today only the legacy model is available.

## D. Statements that are now accurate

- JPG/PNG/WEBP upload with format (content-based) and minimum-resolution (128×128) validation; preview;
  drag-and-drop.
- Grayscale, 224×224, normalisation, Gaussian noise suppression (classical and ResNet18 pipelines), and
  ImageNet mean/std normalised RGB tensors for ResNet18.
- Augmentation: rotation ≤ 10°, small affine transforms, training split only. *Horizontal flipping was
  deliberately excluded* (mirrored handwriting); update the report's augmentation list.
- Metrics (accuracy, precision, recall, F1, confusion matrices) computed per model by
  `src/evaluation/evaluate.py`, once models are trained.
- Results screen with dominant category and confidence, progress bars, plain-language trait descriptions,
  model comparison table, report export and session history.
