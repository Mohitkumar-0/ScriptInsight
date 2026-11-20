"""Heuristic check that an upload looks like a photo or scan of handwriting, run before any model.

The checks are deliberately conservative: they reject images that are blank, that contain a clearly
visible frontal face, or whose dark/light structure does not resemble pen strokes on paper (photographs,
scenery, textures). They cannot tell handwriting from printed text, diagrams or every possible photograph;
see ``docs/HANDWRITING_VALIDATION.md`` for how the thresholds were chosen and the known limitations.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

WORK_SIDE = 1024          # analysis resolution (longest side); larger images are downscaled
FACE_CASCADE = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
EYE_CASCADE = cv2.data.haarcascades + 'haarcascade_eye.xml'

NO_HANDWRITING_MESSAGE = 'No handwriting was detected. Please upload a clear photo or scan of handwritten text.'


@dataclass
class ValidationResult:
    valid: bool
    code: str = 'ok'
    reason: str = ''
    metrics: dict = field(default_factory=dict)


_cascades = {}


def _cascade(path):
    if path not in _cascades:
        _cascades[path] = cv2.CascadeClassifier(path)
    return _cascades[path]


def _faces(gray):
    """Large frontal faces (OpenCV's bundled Haar cascades) as (x, y, w, h, eyes found in the upper half).
    The caller only trusts faces with two eyes, which keeps loops of handwriting from passing as a face."""
    face_detector, eye_detector = _cascade(FACE_CASCADE), _cascade(EYE_CASCADE)
    if face_detector.empty() or eye_detector.empty():
        return []
    gray = cv2.equalizeHist(gray)
    side = min(gray.shape)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(max(48, side // 6),) * 2)
    verified = []
    for x, y, fw, fh in faces:
        upper = gray[y:y + fh // 2 + fh // 10, x:x + fw]
        eyes = eye_detector.detectMultiScale(upper, scaleFactor=1.1, minNeighbors=6,
                                             minSize=(max(8, fw // 10),) * 2, maxSize=(fw // 2,) * 2)
        if len(eyes):
            verified.append((x, y, fw, fh, len(eyes)))
    return verified


def _resize(img):
    h, w = img.shape[:2]
    scale = WORK_SIDE / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)


def measure(gray, rgb=None):
    """Return the measurements the decision is based on (also useful for threshold tuning)."""
    g = _resize(gray)
    h, w = g.shape
    m = {}
    lo, hi = np.percentile(g, [1, 99])
    m['contrast'] = float(hi - lo)
    m['std'] = float(g.std())

    # Paper is the majority tone; flip light-on-dark images so ink is always darker than the background.
    m['inverted'] = bool(np.median(g) < 110)
    if m['inverted']:
        g = 255 - g
    k = max(15, (min(h, w) // 25) | 1)
    background = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    background = cv2.medianBlur(background, 5)
    darkness = cv2.subtract(background, g)
    otsu, _ = cv2.threshold(darkness, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m['ink_threshold'] = float(max(25.0, otsu))
    ink = (darkness > m['ink_threshold']).astype(np.uint8)
    m['ink_ratio'] = float(ink.mean())
    # Ruled/margin lines join a whole page of writing into one component; remove long straight runs.
    rules = cv2.bitwise_or(
        cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, w // 12), 1))),
        cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, h // 12)))))
    ink = cv2.subtract(ink, cv2.dilate(rules, np.ones((3, 3), np.uint8)))
    m['paper_ratio'] = float((darkness < 12).mean())

    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    comp_h, comp_w, area = stats[1:, cv2.CC_STAT_HEIGHT], stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_AREA]
    fill = area / np.maximum(comp_h * comp_w, 1)
    char = ((area >= 12) & (comp_h >= max(4, 0.006 * h)) & (comp_h <= 0.25 * h) & (comp_w <= 0.5 * w)
            & (fill < 0.75))
    m['components'] = int(n - 1)
    m['char_components'] = int(char.sum())
    total_ink = max(int(area.sum()), 1)
    m['char_ink_frac'] = float(area[char].sum() / total_ink)
    left, top = stats[1:, cv2.CC_STAT_LEFT], stats[1:, cv2.CC_STAT_TOP]
    inner = (left > 0) & (top > 0) & (left + comp_w < w) & (top + comp_h < h)
    m['inner_char_ink_frac'] = float(area[char & inner].sum() / max(int(area[inner].sum()), 1))
    if char.sum() >= 2:
        cx, cy = left[char] + comp_w[char] / 2, top[char] + comp_h[char] / 2
        cw, ch = comp_w[char].astype(float), comp_h[char].astype(float)
        # Writing sits on lines: most stroke components (letters or joined-up words) have a neighbour at the
        # same height within a short gap. Checked along both axes so a sideways photo is not rejected.
        dx, dy = np.abs(cx[:, None] - cx[None, :]), np.abs(cy[:, None] - cy[None, :])
        gap_x = dx - (cw[:, None] + cw[None, :]) / 2
        gap_y = dy - (ch[:, None] + ch[None, :]) / 2
        size = np.maximum(ch[:, None], ch[None, :])
        size_t = np.maximum(cw[:, None], cw[None, :])
        np.fill_diagonal(gap_x, np.inf)
        np.fill_diagonal(gap_y, np.inf)
        m['aligned_frac'] = float(max(((dy < 0.5 * size) & (gap_x < 2 * size)).any(axis=1).mean(),
                                      ((dx < 0.5 * size_t) & (gap_y < 2 * size_t)).any(axis=1).mean()))
        ca = area[char].astype(float)
        m['size_cv'] = float(ca.std() / ca.mean())
    else:
        m['aligned_frac'] = m['size_cv'] = 0.0

    if char.any():
        dist = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
        char_mask = np.isin(labels, np.flatnonzero(char) + 1)
        # Stroke thickness ~ 2 x mean distance-to-background over the ridge (local maxima) of the strokes.
        ridge = char_mask & (dist >= cv2.dilate(dist, np.ones((3, 3), np.uint8)))
        m['stroke_width'] = float(2 * dist[ridge].mean()) if ridge.any() else 0.0
        m['char_height'] = float(np.median(comp_h[char]))
        m['char_fill'] = float(np.median(fill[char]))
    else:
        m['stroke_width'] = m['char_height'] = m['char_fill'] = 0.0
    m['stroke_ratio'] = m['stroke_width'] / m['char_height'] if m['char_height'] else 1.0

    if rgb is not None:
        # Saturation of the background, ignoring dark pixels whose hue is mostly sensor noise (so a
        # light-on-dark page such as a blackboard is not penalised).
        hsv = cv2.cvtColor(_resize(rgb), cv2.COLOR_RGB2HSV)
        lit = (ink == 0) & (hsv[..., 2] >= 80)
        m['paper_saturation'] = float(np.median(hsv[..., 1][lit])) if lit.mean() > 0.2 else 0.0
    else:
        m['paper_saturation'] = 0.0
    faces = _faces(_resize(gray))
    m['faces'] = len(faces)
    m['faces_two_eyes'] = sum(f[4] >= 2 for f in faces)
    m['face_frac'] = max((f[2] * f[3] / (h * w) for f in faces if f[4] >= 2), default=0.0)
    return m


# (metric, comparison, (threshold for 1 point, threshold for 2 points)). An image is rejected when the
# points add up to REJECT_POINTS, so no single borderline measurement can reject a page of handwriting.
SUSPICION_RULES = (
    ('paper_ratio', 'below', (0.65, 0.35)),          # little plain background: photo or texture
    ('inner_char_ink_frac', 'below', (0.75, 0.5)),   # dark areas are mostly large blobs, not pen strokes
    ('aligned_frac', 'below', (0.7, 0.4)),           # strokes do not sit side by side along lines
    ('paper_saturation', 'above', (90, None)),       # strongly coloured background (never decisive alone)
    ('char_components', 'below', (30, 15)),          # too few stroke-sized marks
    ('size_cv', 'below', (0.3, 0.1)),                # identical repeated shapes (patterns, not writing)
)
REJECT_POINTS = 2
MIN_CONTRAST = 30
DOMINANT_FACE_FRACTION = 0.15


def suspicion_points(m):
    points = {}
    for key, direction, (one, two) in SUSPICION_RULES:
        value = m[key]
        hit = (lambda t: value < t) if direction == 'below' else (lambda t: value > t)
        if two is not None and hit(two):
            points[key] = 2
        elif hit(one):
            points[key] = 1
    return points


def validate_handwriting(gray, rgb=None):
    """Decide whether a decoded image plausibly shows handwriting. ``gray`` is uint8 H x W; ``rgb`` (optional,
    uint8 H x W x 3) enables the background-colour check."""
    m = measure(gray, rgb)
    if m['contrast'] < MIN_CONTRAST or m['std'] < 5:
        return ValidationResult(False, 'blank_image', 'The image is blank or almost uniform; no writing is visible.', m)
    points = suspicion_points(m)
    m['suspicion_points'] = points
    # A face (with both eyes found) rejects the image when it fills much of the frame or the rest of the image
    # also looks photographic; on a clean page of writing a lone detection is treated as a false alarm.
    if m['faces_two_eyes'] and (m['face_frac'] > DOMINANT_FACE_FRACTION or points):
        return ValidationResult(False, 'face_detected', 'The image appears to show a face or person, not handwriting.', m)
    if sum(points.values()) >= REJECT_POINTS:
        return ValidationResult(False, 'no_handwriting',
                                'No pen strokes arranged like written lines were found; it looks like a photograph '
                                'or picture rather than handwriting.', m)
    return ValidationResult(True, 'ok', 'Handwriting-like strokes detected.', m)
