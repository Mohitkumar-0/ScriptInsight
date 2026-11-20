"""HiEnWrite loader, validation and splitting, on a SYNTHETIC dataset in the HiEnWrite layout."""
import csv
import json

import numpy as np
import pytest

from src.data import hienwrite as hw


def _rows(manifest):
    return {r['image']: r for r in hw.read_manifest(manifest)}


def test_target_mapping_is_explicit_and_complete():
    assert hw.TARGETS == ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness']
    assert hw.TARGET_COLUMNS['Extraversion'] == 'Extroversion'
    assert hw.TARGET_COLUMNS['Openness'] == 'Openness to Experience'


def test_csv_schema_validation(tmp_path):
    good = 'image,Extroversion,Agreeableness,Conscientiousness,Neuroticism,Openness to Experience\n'
    path = tmp_path / 'labels.csv'
    path.write_text('image,Extroversion,Agreeableness\n0e.jpeg,1,2\n')
    with pytest.raises(hw.DatasetError, match='missing required columns'):
        hw.read_csv_rows(path)
    path.write_text(good + '0e.jpeg,10,20,30,abc,25\n')
    with pytest.raises(hw.DatasetError, match='Neuroticism.*not a number'):
        hw.read_csv_rows(path)
    path.write_text(good + '0e.jpeg,10,20,30,-1,25\n')
    with pytest.raises(hw.DatasetError, match='non-negative integer'):
        hw.read_csv_rows(path)
    path.write_text(good + '0e.jpeg,10,20,30,12.5,25\n')
    with pytest.raises(hw.DatasetError, match='non-negative integer'):
        hw.read_csv_rows(path)
    path.write_text(good)
    with pytest.raises(hw.DatasetError, match='no data rows'):
        hw.read_csv_rows(path)
    with pytest.raises(hw.DatasetError, match='not found'):
        hw.read_csv_rows(tmp_path / 'absent.csv')
    path.write_text(good + '12h.jpeg,10,20,30,15,24\n')
    row = hw.read_csv_rows(path)[0]
    assert (row['participant'], row['language']) == ('12', 'h')
    assert row['targets'] == {'Extraversion': 10, 'Agreeableness': 20, 'Conscientiousness': 30, 'Neuroticism': 15,
                              'Openness': 24}


def test_image_path_resolution_is_safe(hienwrite_root, tmp_path):
    images = hienwrite_root / 'images'
    (tmp_path / 'outside.jpeg').write_bytes(b'x')
    index = hw.index_images(images)
    path, reason = hw.resolve_image(images, index, '5e.jpeg')  # stored in a sub-folder
    assert reason is None and path == (images / 'sub' / '5e.jpeg').resolve()
    for name in ('../outside.jpeg', 'sub/5e.jpeg', '..', '/etc/passwd'):
        assert hw.resolve_image(images, index, name) == (None, 'unsafe_image_reference')
    assert hw.resolve_image(images, index, '1h.jpeg') == (None, 'missing_file')
    assert hw.resolve_image(images, index, 'notes.txt') == (None, 'unsupported_extension')
    (images / 'sub' / '0e.jpeg').write_bytes((images / '0e.jpeg').read_bytes())
    assert hw.resolve_image(images, hw.index_images(images), '0e.jpeg') == (None, 'ambiguous_file_name')


def test_validation_excludes_and_counts_bad_files(hienwrite_root, tmp_path):
    manifest, report = hw.validate(hienwrite_root, tmp_path / 'v', n_jobs=1)
    rows = _rows(manifest)
    c = report['counts']
    assert c['csv_rows'] == 48 and c['image_files_on_disk'] == 47 and c['missing_image_files'] == 1
    assert rows['1h.jpeg']['reason'] == 'missing_file'
    assert rows['2e.jpeg']['reason'] == 'unreadable_image:empty_file'
    assert rows['3e.jpeg']['reason'] == 'unreadable_image:corrupt_image'
    assert rows['4h.jpeg']['reason'].startswith('too_small')
    # identical image filed under two participants with different labels: both excluded, never guessed
    assert rows['6e.jpeg']['reason'] == rows['7e.jpeg']['reason'] == 'duplicate_image_conflicting_labels'
    # identical image twice for one participant with agreeing labels: first row kept
    assert rows['8e.jpeg']['status'] == 'usable'
    assert (rows['8h.jpeg']['reason'], rows['8h.jpeg']['duplicate_of']) == ('duplicate_image', '8e.jpeg')
    assert rows['5e.jpeg']['status'] == 'usable' and rows['5e.jpeg']['relative_path'] == 'images/sub/5e.jpeg'
    assert c['usable_samples'] == 48 - 7 == sum(r['status'] == 'usable' for r in rows.values())
    assert c['unreadable_images'] == 2 and c['too_small_images'] == 1 and c['duplicate_images_excluded'] == 3
    assert (tmp_path / 'v' / 'validation_report.md').exists()
    assert json.loads((tmp_path / 'v' / 'validation_report.json').read_text())['counts'] == c


def test_constant_openness_in_hindi_rows_is_masked(hienwrite_root, tmp_path):
    manifest, report = hw.validate(hienwrite_root, tmp_path / 'v', n_jobs=1)
    assert report['constant_targets_masked'] == [{'language': 'h', 'target': 'Openness',
                                                  'csv_column': 'Openness to Experience', 'value': 24, 'n_rows': 24}]
    rows = _rows(manifest)
    assert rows['0h.jpeg']['mask']['Openness'] is False and rows['0e.jpeg']['mask']['Openness'] is True
    assert all(rows['0h.jpeg']['mask'][t] for t in hw.TARGETS if t != 'Openness')
    assert report['target_distributions_usable']['Openness']['n'] == sum(
        r['status'] == 'usable' and r['language'] == 'e' for r in rows.values())


def test_groups_merge_participants_that_share_an_image(hienwrite_root, tmp_path):
    manifest, _ = hw.validate(hienwrite_root, tmp_path / 'v', n_jobs=1)
    rows = _rows(manifest)
    assert rows['6h.jpeg']['group'] == rows['7h.jpeg']['group'] == 'p6'
    assert rows['0e.jpeg']['group'] == rows['0h.jpeg']['group'] == 'p0'


def test_split_is_grouped_reproducible_and_leak_free(hienwrite_root, tmp_path):
    manifest, _ = hw.validate(hienwrite_root, tmp_path / 'v', n_jobs=1)
    split1, meta = hw.make_splits(manifest, tmp_path / 's1', seed=3)
    split2, _ = hw.make_splits(manifest, tmp_path / 's2', seed=3)
    split3, _ = hw.make_splits(manifest, tmp_path / 's3', seed=4)
    assert split1.read_text() == split2.read_text()
    assert split1.read_text() != split3.read_text()
    with split1.open() as f:
        rows = list(csv.DictReader(f))
    for key in ('group', 'participant', 'sha256'):
        seen = {}
        for r in rows:
            seen.setdefault(r[key], set()).add(r['split'])
        assert all(len(s) == 1 for s in seen.values()), key  # never in two splits
    assert meta['n_samples'] == len(rows) == sum(meta['counts'][s]['samples'] for s in ('train', 'val', 'test'))
    assert all(meta['counts'][s]['groups'] >= 1 for s in ('train', 'val', 'test'))
    assert meta['seed'] == 3 and meta['leakage_checks']['groups_in_multiple_splits'] == 0


def test_split_groups_ratios_and_errors():
    groups = [f'p{i}' for i in range(100)]
    assignment = hw.split_groups(groups, seed=1)
    counts = {s: sum(v == s for v in assignment.values()) for s in ('train', 'val', 'test')}
    assert counts == {'train': 70, 'val': 15, 'test': 15}
    with pytest.raises(ValueError, match='at least 3 groups'):
        hw.split_groups(['a', 'b'])
    with pytest.raises(ValueError, match='sum to 1'):
        hw.split_groups(groups, ratios=(0.5, 0.5, 0.5))


def test_read_split_manifest_targets_and_mask(hienwrite_split, hienwrite_root, tmp_path):
    split_manifest, _ = hienwrite_split
    samples, meta = hw.read_split_manifest(split_manifest)
    Y, M = hw.stack_targets(samples)
    assert Y.shape == M.shape == (len(samples), 5) and meta['task'] == 'regression'
    hindi = np.array([s['language'] == 'h' for s in samples])
    assert np.all(np.isnan(Y[hindi, 4])) and not M[hindi, 4].any()   # Openness masked for Hindi
    assert np.all(np.isfinite(Y[:, :4])) and M[:, :4].all()
    stats = hw.target_stats([s for s in samples if s['split'] == 'train'])
    assert stats['Openness']['n'] == int(M[[s['split'] == 'train' for s in samples], 4].sum())
    (hienwrite_root / 'images' / '0e.jpeg').unlink()
    with pytest.raises(hw.DatasetError, match='missing'):
        hw.read_split_manifest(split_manifest)


def test_missing_manifest_gives_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match='python -m src.data.hienwrite'):
        hw.read_split_manifest(tmp_path / 'split_manifest.csv')
    with pytest.raises(FileNotFoundError, match='validate'):
        hw.make_splits(tmp_path / 'data_manifest.csv', tmp_path / 's')
