"""HiEnWrite dataset: validation, data manifest and participant-grouped splits for five-target regression.

HiEnWrite (Checker, Yadav & Katarya, ACM TALLIP 2025, doi:10.1145/3756010; CC BY-NC 4.0) ships
``data/merged_data.csv`` (one row per image, five integer Big Five scores, Age, Gender) and ``images/``.
It is kept separate from the HWxPI / ``labels.csv`` loader in ``dataset.py``: the labels here are numerical
scores, not 0/1 poles, and are never thresholded or converted.

What this module does (all decisions are recorded in the reports it writes):

* Schema: requires ``image`` and the five score columns; every score must be a finite integer >= 0.
* Paths: an ``image`` value must be a bare file name (no directories, no ``..``); it is looked up in a
  recursive index of ``images/`` and the resolved file must lie inside that folder.
* Images: every file is decoded with the same code used for uploads (EXIF orientation applied). Missing,
  empty, undecodable and too-small (< MIN_IMAGE_SIDE) files are excluded with a reason, never replaced.
* Constant targets: if a score is identical for every row of one language (n >= 20), it carries no
  information about that subset and is MASKED (treated as missing) for those rows. In HiEnWrite v1.0.0
  Openness is 24 for all Hindi rows.
* Duplicates (same SHA-256, or the same photograph re-encoded: dHash <= 6 bits): if all copies have the same labels, the first row in CSV order is kept;
  if the labels conflict, every copy is excluded, because the correct label cannot be known.
* Groups: the dataset has no documented writer-id column. Image ids look like ``<number><e|h>.jpeg``; the
  e/h rows of one number share four of five scores, Age and Gender in every case, and the paper reports
  ~800 images from ~400 authors, so the number is treated as an INFERRED participant id. Participants
  that share an identical or near-identical image are merged into one group. Splits are made over groups,
  so both language samples of a participant always land in the same split.

Usage:
    python -m src.data.hienwrite validate --data-root ~/Downloads/HiEnWrite-Dataset-main
    python -m src.data.hienwrite split
"""
import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import config
from src.preprocessing.image_processor import ImageValidationError, decode_image_file

CSV_RELATIVE_PATH = Path('data') / 'merged_data.csv'
IMAGES_DIRNAME = 'images'
# Internal target name -> original CSV column. Internal names match config.TRAITS so the UI can label them.
TARGET_COLUMNS = {
    'Extraversion': 'Extroversion',
    'Agreeableness': 'Agreeableness',
    'Conscientiousness': 'Conscientiousness',
    'Neuroticism': 'Neuroticism',
    'Openness': 'Openness to Experience',
}
TARGETS = list(TARGET_COLUMNS)
IMAGE_ID_RE = re.compile(r'^(?P<participant>\d+)(?P<language>[eh])\.(?:jpe?g|png)$', re.IGNORECASE)
LANGUAGES = {'e': 'English', 'h': 'Hindi'}
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png'}
CONSTANT_TARGET_MIN_ROWS = 20
NEAR_DUPLICATE_BITS = 6
DEFAULT_RATIOS = (0.70, 0.15, 0.15)
MANIFEST_NAME = 'data_manifest.csv'
SPLIT_MANIFEST_NAME = 'split_manifest.csv'


class DatasetError(ValueError):
    """The dataset files do not match the expected HiEnWrite schema."""


# ----------------------------------------------------------------------------------------------- loading
def read_csv_rows(csv_path):
    """Parse and validate the label CSV. Returns a list of row dicts; raises DatasetError on bad data."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise DatasetError(f'Label file not found: {csv_path}')
    with csv_path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        missing = [c for c in ['image', *TARGET_COLUMNS.values()] if c not in fields]
        if missing:
            raise DatasetError(f'{csv_path.name} is missing required columns {missing}; found {fields}.')
        rows = []
        for line, raw in enumerate(reader, start=2):
            image = (raw['image'] or '').strip()
            if not image:
                raise DatasetError(f'{csv_path.name} line {line}: empty "image" value.')
            targets = {}
            for name, column in TARGET_COLUMNS.items():
                value = (raw[column] or '').strip()
                try:
                    number = float(value)
                except ValueError:
                    raise DatasetError(f'{csv_path.name} line {line} ({image}): {column} = {value!r} is not a number.')
                if not math.isfinite(number) or number < 0 or number != int(number):
                    raise DatasetError(f'{csv_path.name} line {line} ({image}): {column} = {value!r} is not a '
                                       'non-negative integer score.')
                targets[name] = int(number)
            match = IMAGE_ID_RE.match(image)
            rows.append({
                'row_index': len(rows), 'line': line, 'image': image, 'targets': targets,
                'participant': str(int(match['participant'])) if match else None,
                'language': match['language'].lower() if match else None,
                'age': (raw.get('Age') or '').strip(), 'gender': (raw.get('Gender') or '').strip(),
            })
    if not rows:
        raise DatasetError(f'{csv_path.name} has no data rows.')
    return rows


def index_images(images_dir):
    """Recursive index: file name -> sorted list of paths (more than one path means the name is ambiguous)."""
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        raise DatasetError(f'Image folder not found: {images_dir}')
    index = defaultdict(list)
    for path in sorted(images_dir.rglob('*')):
        if path.is_file() and not path.name.startswith('.'):
            index[path.name].append(path)
    return dict(index)


def resolve_image(images_dir, index, name):
    """Return (path, None) or (None, reason). Only bare file names inside ``images_dir`` are accepted."""
    if name != Path(name).name or name in {'.', '..'} or '\\' in name or '\0' in name:
        return None, 'unsafe_image_reference'
    if Path(name).suffix.lower() not in IMAGE_SUFFIXES:
        return None, 'unsupported_extension'
    paths = index.get(name, [])
    if not paths:
        return None, 'missing_file'
    if len(paths) > 1:
        return None, 'ambiguous_file_name'
    root = Path(images_dir).resolve()
    path = paths[0].resolve()
    if root not in path.parents:
        return None, 'unsafe_image_reference'
    return path, None


def _dhash_bits(gray, size=16):
    import cv2
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    return np.packbits((small[:, 1:] > small[:, :-1]).flatten())


def inspect_image(path):
    """Decode one image with the upload decoder. Returns a dict with hash, size and an error code if any."""
    data = Path(path).read_bytes()
    info = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data), 'width': None, 'height': None,
            'format': None, 'error': None, 'dhash': None}
    try:
        decoded = decode_image_file(path, min_side=1)
    except ImageValidationError as exc:
        info['error'] = exc.code
        return info
    info.update(width=decoded.width, height=decoded.height, format=decoded.format,
                dhash=_dhash_bits(decoded.gray).tobytes().hex())
    return info


# ------------------------------------------------------------------------------------------ validation
def find_constant_targets(rows, min_rows=CONSTANT_TARGET_MIN_ROWS):
    """Targets whose value never varies within one language subset (no information; masked for training)."""
    found = []
    for language in sorted({r['language'] for r in rows if r['language']}):
        subset = [r for r in rows if r['language'] == language]
        for name in TARGETS:
            values = {r['targets'][name] for r in subset}
            if len(subset) >= min_rows and len(values) == 1:
                found.append({'language': language, 'target': name, 'csv_column': TARGET_COLUMNS[name],
                              'value': values.pop(), 'n_rows': len(subset)})
    return found


class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb, key=_participant_key)] = min(ra, rb, key=_participant_key)


def _participant_key(p):
    return (0, int(p), p) if p.isdigit() else (1, 0, p)


def _labels_agree(a, b):
    """True if two rows have the same score for every target that is valid (unmasked) in both."""
    return all(a['targets'][t] == b['targets'][t] for t in TARGETS if a['mask'][t] and b['mask'][t])


def _near_duplicate_pairs(rows, max_bits):
    usable = [r for r in rows if r.get('dhash')]
    if len(usable) < 2:
        return []
    hashes = np.array([np.frombuffer(bytes.fromhex(r['dhash']), dtype=np.uint8) for r in usable])
    popcount = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1)
    pairs = []
    for i in range(len(usable) - 1):
        distances = popcount[np.bitwise_xor(hashes[i + 1:], hashes[i])].sum(1)
        for j in np.nonzero(distances <= max_bits)[0] + i + 1:
            if usable[i]['sha256'] != usable[j]['sha256']:
                pairs.append((usable[i], usable[j], int(distances[j - i - 1])))
    return pairs


def _distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {'n': 0}
    hist, edges = np.histogram(values, bins=8)
    return {'n': int(len(values)), 'mean': round(float(values.mean()), 3), 'std': round(float(values.std(ddof=1)), 3)
            if len(values) > 1 else 0.0, 'min': float(values.min()), 'q25': float(np.percentile(values, 25)),
            'median': float(np.median(values)), 'q75': float(np.percentile(values, 75)), 'max': float(values.max()),
            'histogram': {'counts': hist.tolist(), 'edges': [round(float(e), 2) for e in edges]}}


def target_distributions(rows):
    """Per-target distribution over the given rows, counting only valid (unmasked) values."""
    return {t: _distribution([r['targets'][t] for r in rows if r['mask'][t]]) for t in TARGETS}


def validate(data_root, out_dir=config.HIENWRITE_VALIDATION_DIR, min_side=None, near_duplicate_bits=NEAR_DUPLICATE_BITS,
             n_jobs=-1):
    """Validate the dataset and write ``data_manifest.csv`` + ``validation_report.{json,md}``."""
    data_root = Path(data_root).expanduser().resolve()
    min_side = config.MIN_IMAGE_SIDE if min_side is None else min_side
    rows = read_csv_rows(data_root / CSV_RELATIVE_PATH)
    images_dir = data_root / IMAGES_DIRNAME
    index = index_images(images_dir)

    # Rows whose id does not follow <number><e|h> cannot be assigned a participant: exclude, do not guess.
    for r in rows:
        r.update(status='usable', reason='', duplicate_of='', path=None, relative_path='')
        if r['participant'] is None:
            r.update(status='excluded', reason='unrecognised_image_id')

    image_counts = Counter(r['image'] for r in rows)
    for r in rows:
        if image_counts[r['image']] > 1 and r['status'] == 'usable':
            r.update(status='excluded', reason='image_listed_more_than_once')

    constant = find_constant_targets([r for r in rows if r['language']])
    masked = {(c['language'], c['target']) for c in constant}
    for r in rows:
        r['mask'] = {t: (r['language'], t) not in masked for t in TARGETS}

    for r in rows:
        if r['status'] != 'usable':
            continue
        path, reason = resolve_image(images_dir, index, r['image'])
        if reason:
            r.update(status='excluded', reason=reason)
        else:
            r['path'] = path
            r['relative_path'] = path.relative_to(data_root).as_posix()

    from joblib import Parallel, delayed
    to_inspect = [r for r in rows if r['path'] is not None]
    infos = Parallel(n_jobs=n_jobs)(delayed(inspect_image)(r['path']) for r in to_inspect)
    for r, info in zip(to_inspect, infos):
        r.update(info)
        if info['error']:
            r.update(status='excluded', reason=f"unreadable_image:{info['error']}")
        elif min(info['width'], info['height']) < min_side:
            r.update(status='excluded', reason=f'too_small(<{min_side}px)')

    # Duplicate clusters: decodable images linked by an identical SHA-256 or a near-identical dHash (the same
    # photograph re-encoded or colour-adjusted; every such pair in v1.0.0 was checked visually).
    decodable = [r for r in to_inspect if not r['error']]
    near_pairs = _near_duplicate_pairs(decodable, near_duplicate_bits)
    image_uf = _UnionFind()
    by_hash = defaultdict(list)
    for r in decodable:
        image_uf.find(r['image'])
        by_hash[r['sha256']].append(r)
    for cluster in by_hash.values():
        for r in cluster[1:]:
            image_uf.union(cluster[0]['image'], r['image'])
    for a, b, _ in near_pairs:
        image_uf.union(a['image'], b['image'])
    clusters = defaultdict(list)
    for r in decodable:
        clusters[image_uf.find(r['image'])].append(r)

    duplicate_clusters = []
    for cluster in clusters.values():
        if len(cluster) < 2:
            continue
        cluster.sort(key=lambda r: r['row_index'])
        exact = len({r['sha256'] for r in cluster}) == 1
        agree = all(_labels_agree(a, b) for i, a in enumerate(cluster) for b in cluster[i + 1:])
        candidates = [r for r in cluster if r['status'] == 'usable']
        keep = candidates[0] if agree and candidates else None
        for r in cluster:
            if r['status'] == 'usable' and r is not keep:
                r.update(status='excluded', reason='duplicate_image' if agree else 'duplicate_image_conflicting_labels',
                         duplicate_of=keep['image'] if keep else '')
        duplicate_clusters.append({'images': [r['image'] for r in cluster], 'byte_identical': exact,
                                   'labels_agree': agree, 'kept': keep['image'] if keep else None})

    # Groups: inferred participant, merged through any shared exact or near-duplicate image.
    uf = _UnionFind()
    for r in rows:
        if r['participant'] is not None:
            uf.find(r['participant'])
    for cluster in clusters.values():
        for r in cluster[1:]:
            uf.union(cluster[0]['participant'], r['participant'])
    for r in rows:
        r['group'] = f"p{uf.find(r['participant'])}" if r['participant'] is not None else ''

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / MANIFEST_NAME
    write_manifest(manifest_path, rows)

    usable = [r for r in rows if r['status'] == 'usable']
    reasons = Counter(r['reason'] for r in rows if r['status'] != 'usable')
    referenced = {r['image'] for r in rows}
    sizes = np.array([[r['width'], r['height']] for r in to_inspect if not r['error']], dtype=float)
    pair_counts = Counter(r['participant'] for r in rows)
    same_except = sum(
        1 for p, n in pair_counts.items() if n == 2 and len({
            (tuple(r['targets'][t] for t in TARGETS if t != 'Openness'), r['age'], r['gender'])
            for r in rows if r['participant'] == p}) == 1)
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'data_root': str(data_root),
        'csv': str(CSV_RELATIVE_PATH), 'csv_sha256': _file_sha256(data_root / CSV_RELATIVE_PATH),
        'csv_columns_used': {'image': 'image', **TARGET_COLUMNS},
        'min_image_side': min_side,
        'counts': {
            'csv_rows': len(rows),
            'unique_image_references': len(referenced),
            'image_files_on_disk': sum(len(v) for v in index.values()),
            'csv_rows_matched_to_a_file': len(to_inspect),
            'missing_image_files': reasons.get('missing_file', 0),
            'files_on_disk_not_in_csv': sorted(set(index) - referenced),
            'unreadable_images': sum(v for k, v in reasons.items() if k.startswith('unreadable_image')),
            'too_small_images': sum(v for k, v in reasons.items() if k.startswith('too_small')),
            'duplicate_images_excluded': reasons.get('duplicate_image', 0)
            + reasons.get('duplicate_image_conflicting_labels', 0),
            'usable_samples': len(usable),
            'usable_by_language': {LANGUAGES[k]: v for k, v in sorted(Counter(r['language'] for r in usable).items())},
            'participants_in_csv': len(pair_counts),
            'participants_usable': len({r['participant'] for r in usable}),
            'groups_usable': len({r['group'] for r in usable}),
        },
        'exclusions_by_reason': dict(sorted(reasons.items())),
        'excluded_rows': [{'image': r['image'], 'reason': r['reason'], 'duplicate_of': r['duplicate_of']}
                          for r in rows if r['status'] != 'usable'],
        'duplicate_csv_rows': sum(1 for n in image_counts.values() if n > 1),
        'duplicate_clusters': duplicate_clusters,
        'near_duplicate_pairs': [{'a': a['image'], 'b': b['image'], 'dhash_bits': d,
                                  'same_participant': a['participant'] == b['participant']} for a, b, d in near_pairs],
        'constant_targets_masked': constant,
        'writer_ids': {
            'documented_column': False,
            'strategy': 'inferred participant = numeric prefix of the image id (<number><e|h>.jpeg), merged '
                        'through shared exact / near-duplicate images',
            'evidence': {
                'participants_with_two_rows': sum(1 for n in pair_counts.values() if n == 2),
                'pairs_identical_on_E_A_C_N_age_gender': same_except,
                'paper_abstract': 'handwritten samples collected from over 400 authors; approximately 800 images',
            },
            'limitation': 'Writer identity is not documented in the CSV or readme. If one person contributed '
                          'under several numbers, their samples could fall in different splits.',
        },
        'image_size': {'width': _distribution(sizes[:, 0]) if len(sizes) else {},
                       'height': _distribution(sizes[:, 1]) if len(sizes) else {}},
        'target_ranges_all_rows': {t: [min(r['targets'][t] for r in rows), max(r['targets'][t] for r in rows)]
                                   for t in TARGETS},
        'target_distributions_usable': target_distributions(usable),
        'target_distributions_usable_by_language': {
            LANGUAGES[l]: target_distributions([r for r in usable if r['language'] == l]) for l in sorted(LANGUAGES)},
        'gender_counts_usable': dict(Counter(r['gender'] for r in usable)),
        'age_values_usable': dict(sorted(Counter(r['age'] for r in usable).items())),
    }
    (out_dir / 'validation_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (out_dir / 'validation_report.md').write_text(_report_markdown(report), encoding='utf-8')
    return manifest_path, report


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


MANIFEST_FIELDS = ['row_index', 'image', 'relative_path', 'participant', 'language', 'group', 'status', 'reason',
                   'duplicate_of', 'sha256', 'width', 'height', *TARGETS, *[f'mask_{t}' for t in TARGETS], 'age', 'gender']


def write_manifest(path, rows, extra_fields=()):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([*MANIFEST_FIELDS, *extra_fields])
        for r in rows:
            writer.writerow([r['row_index'], r['image'], r['relative_path'], r['participant'] or '', r['language'] or '',
                             r['group'], r['status'], r['reason'], r['duplicate_of'], r.get('sha256') or '',
                             r.get('width') or '', r.get('height') or '', *[r['targets'][t] for t in TARGETS],
                             *[int(r['mask'][t]) for t in TARGETS], r['age'], r['gender'],
                             *[r[k] for k in extra_fields]])


def read_manifest(path):
    rows = []
    with Path(path).open(newline='', encoding='utf-8') as f:
        for raw in csv.DictReader(f):
            row = dict(raw)
            row['row_index'] = int(raw['row_index'])
            row['targets'] = {t: int(raw[t]) for t in TARGETS}
            row['mask'] = {t: raw[f'mask_{t}'] == '1' for t in TARGETS}
            rows.append(row)
    return rows


def _report_markdown(report):
    c = report['counts']
    lines = [
        '# HiEnWrite validation report', '',
        f"Generated {report['created_utc']} from `{report['data_root']}` (CSV sha256 `{report['csv_sha256'][:16]}`).", '',
        '| Count | Value |', '|---|---|',
        *[f'| {k.replace("_", " ")} | {v if not isinstance(v, list) else len(v)} |' for k, v in c.items()
          if not isinstance(v, dict)],
        f"| usable by language | {', '.join(f'{k}: {v}' for k, v in c['usable_by_language'].items())} |", '',
        '## Exclusions', '', '| Reason | Rows |', '|---|---|',
        *[f'| {k} | {v} |' for k, v in report['exclusions_by_reason'].items()], '',
        'Excluded images: ' + ', '.join(f"`{e['image']}` ({e['reason']})" for e in report['excluded_rows']), '',
        '## Masked targets', '',
        *[f"- **{m['target']}** (`{m['csv_column']}`) is {m['value']} for all {m['n_rows']} "
          f"{LANGUAGES[m['language']]} rows; treated as missing for those rows." for m in report['constant_targets_masked']],
        '', '## Writer identity', '', report['writer_ids']['strategy'] + '.', '',
        f"Evidence: {json.dumps(report['writer_ids']['evidence'])}", '', report['writer_ids']['limitation'], '',
        '## Duplicate images', '',
        'Clusters of byte-identical files or near-identical photographs (dHash <= '
        f'{NEAR_DUPLICATE_BITS} of 256 bits; {len(report["near_duplicate_pairs"])} such pairs). Participants that share '
        'a cluster are also merged into one split group.', '',
        *[f"- {', '.join(d['images'])} ({'byte-identical' if d['byte_identical'] else 'incl. near-identical'}): labels "
          f"{'agree (kept ' + str(d['kept']) + ')' if d['labels_agree'] else 'CONFLICT (all excluded)'}"
          for d in report['duplicate_clusters']], '',
        '## Target distributions (usable rows, valid values only)', '',
        '| Target | n | mean | std | min | median | max |', '|---|---|---|---|---|---|---|',
        *[f"| {t} | {d['n']} | {d['mean']} | {d['std']} | {d['min']:g} | {d['median']:g} | {d['max']:g} |"
          for t, d in report['target_distributions_usable'].items()], '',
    ]
    return '\n'.join(lines)


# ------------------------------------------------------------------------------------------- splitting
def split_groups(groups, seed=config.SEED, ratios=DEFAULT_RATIOS):
    """Deterministically assign whole groups to train/val/test. Returns {group: split}."""
    if not math.isclose(sum(ratios), 1.0) or any(r <= 0 for r in ratios):
        raise ValueError(f'Split ratios must be positive and sum to 1; got {ratios}.')
    unique = sorted(set(groups), key=lambda g: (len(g), g))
    if len(unique) < 3:
        raise ValueError(f'Need at least 3 groups to split; found {len(unique)}.')
    order = np.random.default_rng(seed).permutation(len(unique))
    n_test = max(1, round(ratios[2] * len(unique)))
    n_val = max(1, round(ratios[1] * len(unique)))
    assignment = {}
    for rank, idx in enumerate(order):
        assignment[unique[idx]] = 'test' if rank < n_test else 'val' if rank < n_test + n_val else 'train'
    return assignment


def make_splits(manifest_path=config.HIENWRITE_VALIDATION_DIR / MANIFEST_NAME, out_dir=config.HIENWRITE_SPLITS_DIR,
                seed=config.SEED, ratios=DEFAULT_RATIOS):
    """Split usable rows of a data manifest by group and write ``split_manifest.csv`` + ``split_meta.json``."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f'Data manifest not found at {manifest_path}. Run: '
                                'python -m src.data.hienwrite validate --data-root <HiEnWrite folder>')
    report_path = manifest_path.parent / 'validation_report.json'
    report = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {}
    rows = [r for r in read_manifest(manifest_path) if r['status'] == 'usable']
    if not rows:
        raise ValueError('The data manifest has no usable rows.')
    assignment = split_groups([r['group'] for r in rows], seed, ratios)
    for r in rows:
        r['split'] = assignment[r['group']]

    splits_of_group = defaultdict(set)
    splits_of_participant = defaultdict(set)
    for r in rows:
        splits_of_group[r['group']].add(r['split'])
        splits_of_participant[r['participant']].add(r['split'])
    leaks = [g for g, s in splits_of_group.items() if len(s) > 1] + \
            [p for p, s in splits_of_participant.items() if len(s) > 1]
    hashes = defaultdict(set)
    for r in rows:
        hashes[r['sha256']].add(r['split'])
    leaks += [h for h, s in hashes.items() if len(s) > 1]
    if leaks:
        raise RuntimeError(f'Leakage check failed: {len(leaks)} groups/participants/images appear in several splits.')

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_manifest = out_dir / SPLIT_MANIFEST_NAME
    write_manifest(split_manifest, rows, extra_fields=('split',))
    names = ('train', 'val', 'test')
    meta = {
        'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'dataset': 'HiEnWrite',
        'task': 'regression',
        'data_root': report.get('data_root'),
        'data_manifest': str(manifest_path.resolve()),
        'csv_sha256': report.get('csv_sha256'),
        'dataset_fingerprint': hashlib.sha256('|'.join(sorted(f"{r['image']}:{r['sha256']}" for r in rows))
                                              .encode()).hexdigest()[:16],
        'targets': TARGETS, 'csv_columns': TARGET_COLUMNS,
        'seed': seed, 'ratios': dict(zip(names, ratios)),
        'split_unit': 'group (inferred participant, merged through shared images)',
        'n_samples': len(rows), 'n_groups': len(assignment),
        'counts': {s: {'samples': sum(r['split'] == s for r in rows),
                       'groups': sum(v == s for v in assignment.values()),
                       'by_language': dict(sorted(Counter(r['language'] for r in rows if r['split'] == s).items()))}
                   for s in names},
        'groups': {s: sorted((g for g, v in assignment.items() if v == s), key=lambda g: (len(g), g)) for s in names},
        'target_distributions': {s: target_distributions([r for r in rows if r['split'] == s]) for s in names},
        'masked_targets': report.get('constant_targets_masked', []),
        'leakage_checks': {'groups_in_multiple_splits': 0, 'participants_in_multiple_splits': 0,
                           'identical_images_in_multiple_splits': 0},
        'notes': [
            'Splits are made over groups before any feature fitting or augmentation; scalers, PCA and target '
            'standardisation are fitted on the train split only.',
            'Writer ids are inferred, not documented (see validation_report.md).',
            'Model selection and early stopping use the val split; the test split is read only by evaluation.',
        ],
    }
    (out_dir / 'split_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return split_manifest, meta


def read_split_manifest(path=config.HIENWRITE_SPLITS_DIR / SPLIT_MANIFEST_NAME, data_root=None):
    """Return (samples, meta). Each sample has 'path', 'split', 'group', 'language', 'targets' (float array,
    NaN where masked) and 'mask' (bool array), both in TARGETS order."""
    path = Path(path)
    meta_path = path.parent / 'split_meta.json'
    if not path.exists() or not meta_path.exists():
        raise FileNotFoundError(f'HiEnWrite split manifest not found at {path}. Run: '
                                'python -m src.data.hienwrite validate --data-root <HiEnWrite folder> && '
                                'python -m src.data.hienwrite split')
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    root = Path(data_root or meta['data_root']).expanduser().resolve()
    samples = []
    for r in read_manifest(path):
        image_path = (root / r['relative_path']).resolve()
        if root not in image_path.parents:
            raise DatasetError(f'Manifest path escapes the dataset folder: {r["relative_path"]}')
        if not image_path.is_file():
            raise DatasetError(f'Image listed in the split manifest is missing: {image_path}. '
                               'Pass --data-root if the dataset moved.')
        mask = np.array([r['mask'][t] for t in TARGETS], dtype=bool)
        targets = np.array([r['targets'][t] if r['mask'][t] else np.nan for t in TARGETS], dtype=np.float64)
        samples.append({'image': r['image'], 'path': str(image_path), 'split': r['split'], 'group': r['group'],
                        'participant': r['participant'], 'language': r['language'], 'sha256': r['sha256'],
                        'targets': targets, 'mask': mask})
    return samples, meta


def target_stats(samples):
    """Per-target mean/std/min/max over valid values of the given (training) samples."""
    Y, M = stack_targets(samples)
    stats = {}
    for i, name in enumerate(TARGETS):
        v = Y[M[:, i], i]
        stats[name] = {'n': int(len(v)), 'mean': float(v.mean()), 'std': float(v.std(ddof=1)),
                       'min': float(v.min()), 'max': float(v.max())} if len(v) > 1 else {'n': int(len(v))}
    return stats


def stack_targets(samples):
    """(Y, M): float targets with NaN where masked, and the boolean validity mask, shape (n, 5)."""
    return (np.stack([s['targets'] for s in samples]) if samples else np.zeros((0, len(TARGETS))),
            np.stack([s['mask'] for s in samples]) if samples else np.zeros((0, len(TARGETS)), bool))


# ------------------------------------------------------------------------------------------------- CLI
def main():
    parser = argparse.ArgumentParser(description='Validate HiEnWrite and create participant-grouped splits.')
    sub = parser.add_subparsers(dest='command', required=True)
    v = sub.add_parser('validate', help='Validate CSV + images and write the data manifest and report.')
    v.add_argument('--data-root', default=str(config.HIENWRITE_ROOT))
    v.add_argument('--out', default=str(config.HIENWRITE_VALIDATION_DIR))
    v.add_argument('--min-side', type=int, default=config.MIN_IMAGE_SIDE)
    v.add_argument('--jobs', type=int, default=-1)
    s = sub.add_parser('split', help='Create the train/val/test split manifest from the data manifest.')
    s.add_argument('--manifest', default=str(config.HIENWRITE_VALIDATION_DIR / MANIFEST_NAME))
    s.add_argument('--out', default=str(config.HIENWRITE_SPLITS_DIR))
    s.add_argument('--seed', type=int, default=config.SEED)
    s.add_argument('--ratios', type=float, nargs=3, default=list(DEFAULT_RATIOS), metavar=('TRAIN', 'VAL', 'TEST'))
    args = parser.parse_args()
    try:
        if args.command == 'validate':
            manifest, report = validate(args.data_root, args.out, args.min_side, n_jobs=args.jobs)
            print(f'Data manifest: {manifest}')
            print(json.dumps({k: v for k, v in report['counts'].items()}, indent=2))
            print('Exclusions:', json.dumps(report['exclusions_by_reason']))
            for m in report['constant_targets_masked']:
                print(f"Masked: {m['target']} = {m['value']} for all {m['n_rows']} {LANGUAGES[m['language']]} rows")
        else:
            manifest, meta = make_splits(args.manifest, args.out, args.seed, tuple(args.ratios))
            print(f'Split manifest: {manifest}')
            print(json.dumps(meta['counts'], indent=2))
    except (DatasetError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(f'error: {exc}')


if __name__ == '__main__':
    main()
