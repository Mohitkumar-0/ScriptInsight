# Handwriting-content validation

`src/preprocessing/handwriting_validator.py` runs on every upload to `/api/predict` and `/predict`, after the
image is decoded and before any model is called. If the image is rejected, the response is HTTP 400 and no
model runs:

```json
{"success": false, "code": "invalid_handwriting_image",
 "error": "No handwriting was detected. Please upload a clear photo or scan of handwritten text.",
 "message": "No handwriting was detected. Please upload a clear photo or scan of handwritten text.",
 "reason": "blank_image | face_detected | no_handwriting", "detail": "<specific reason>"}
```

`error` holds the same text as `message`, so existing clients that display `error` keep working. Corrupt,
empty, too-small or unsupported files are still rejected earlier by `decode_image`, using their existing
codes (422/415/413/400).

## Method

Everything is measured at a working size of at most 1024 px on the longest side. Light-on-dark images are
inverted first, so the ink is always darker than the paper.

1. **Blank**: rejected if the 1st–99th percentile contrast is below 30 grey levels or the standard deviation
   is below 5.
2. **Ink extraction**: each pixel is compared with a local paper estimate (morphological closing). Ink is
   whatever is darker than the paper by more than max(25, Otsu). Long horizontal and vertical runs (ruled
   lines, margins) are then removed, and the remaining ink is split into connected components.
3. **Suspicion points.** Each measurement adds 0, 1 or 2 points. An image is rejected at **2 points**, so
   no single borderline measurement can reject a page:

   | Measurement | 1 point | 2 points |
   |---|---|---|
   | Plain-paper share of pixels | < 0.65 | < 0.35 |
   | Share of ink in stroke-sized components (not touching the border) | < 0.75 | < 0.50 |
   | Share of stroke components with a same-line neighbour (either axis) | < 0.70 | < 0.40 |
   | Background HSV saturation (bright pixels only) | > 90 | never |
   | Number of stroke-sized components | < 30 | < 15 |
   | Size variation of components (repeated identical shapes) | < 0.30 | < 0.10 |

4. **Faces**: OpenCV's bundled Haar frontal-face cascade, with a detection confirmed only when **two eyes**
   are found in the upper half of the face. A confirmed face rejects the image if it covers more than 15%
   of the frame or the image also scored at least one suspicion point.

The analysis takes about 25 ms for a 1600×900 photo. No model is downloaded and no new dependency is added.

## Evaluation (thresholds were tuned on the same data)

- **Handwriting.** All 654 decodable HiEnWrite images, plus the 4 sample photos in the project root, were
  accepted. The dataset also contains one face photo (`176h.jpeg`) and one blank image (`293h.jpeg`);
  these were counted as negatives, and both are rejected.
- **Robustness.** Each handwriting image was also tested as grayscale-only, inverted, rotated 90° and
  downscaled to 640 px. Of 3,263 checks, 3 failed (two unusual pages: `227e`, `278e`).
- **Non-handwriting.** 34 images: scikit-image and scikit-learn sample photos, synthetic blanks and noise,
  and the two dataset negatives. 31 were rejected. Three were accepted: `coins` (grey coins on a dark
  background), `moon` (low-texture crater photo) and `page` (a printed book page).

Because the thresholds were tuned on these same images, these figures are optimistic. They are not an
independent accuracy estimate.

## Known limitations

- **Content.** The validator checks stroke and line structure; it does not recognise image content.
  Printed text, typed documents, diagrams, sketches and some photographs with text-like texture can pass.
- **Faces.** Face detection works only for fairly frontal, unobstructed, reasonably large faces. Profile,
  tilted, small or heavily shadowed faces may not be detected. Such images are then judged only by the
  stroke checks, which usually but not always reject photos.
- **Handwriting.** Very sparse writing (a few words), very faint pencil, heavy shadows or a page that fills
  only a small part of a cluttered photo may be rejected.
- **Pretrained classifiers.** An ImageNet classifier (the locally cached torchvision ResNet18 weights) was
  tried as a content classifier and not used. Handwriting pages were labelled "fountain pen", "menu",
  "jigsaw puzzle" and similar. ImageNet has no person/face class, so the portrait photo came out as "reel"
  or "stethoscope". Those weights are also not part of the project.
