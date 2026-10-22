import csv
import json

import pytest

import config
from src.data import dataset
from src.data.hwxpi import convert


def _groups_by_split(samples):
    out = {}
    for s in samples:
        out.setdefault(s['group'], set()).add(s['split'])
    return out


def test_multilabel_split_is_grouped_80_10_10_and_reproducible(multilabel_dataset, tmp_path):
    manifest, meta = dataset.prepare(multilabel_dataset, tmp_path / 's1', seed=7)
    samples, saved_meta = dataset.read_manifest(manifest)
    assert meta['label_mode'] == 'multilabel' and saved_meta['label_names'] == config.TRAITS
    assert meta['grouping'] == 'participant_id' and meta['n_groups'] == 60
    assert all(len(splits) == 1 for splits in _groups_by_split(samples).values())  # no writer in two splits
    groups = {split: len({s['group'] for s in samples if s['split'] == split}) for split in ('train', 'val', 'test')}
    assert groups == {'train': 48, 'val': 6, 'test': 6}
    manifest2, _ = dataset.prepare(multilabel_dataset, tmp_path / 's2', seed=7)
    assert manifest.read_text() == manifest2.read_text()
    manifest3, _ = dataset.prepare(multilabel_dataset, tmp_path / 's3', seed=8)
    assert manifest.read_text() != manifest3.read_text()


def test_split_metadata_records_distribution(multilabel_dataset, tmp_path):
    _, meta = dataset.prepare(multilabel_dataset, tmp_path / 's')
    on_disk = json.loads((tmp_path / 's' / 'split_meta.json').read_text())
    assert on_disk['seed'] == config.SEED and on_disk['cross_split_groups'] == 0
    assert sum(on_disk['distribution'][s]['n'] for s in ('train', 'val', 'test')) == meta['n_samples'] == 120


def test_near_duplicate_copies_stay_in_one_split(multiclass_dataset, tmp_path):
    manifest, meta = dataset.prepare(multiclass_dataset, tmp_path / 's')
    samples, _ = dataset.read_manifest(manifest)
    assert meta['label_mode'] == 'multiclass'
    assert meta['grouping'].startswith('near_duplicate')
    assert meta['n_groups'] == 30  # 60 files = 30 originals x 2 near-identical copies
    assert all(len(splits) == 1 for splits in _groups_by_split(samples).values())
    assert any('ONE category per image' in w for w in meta['warnings'])


def test_too_few_groups_raises(tmp_path, multiclass_dataset):
    for trait_dir in multiclass_dataset.iterdir():
        for extra in sorted(trait_dir.iterdir())[2:]:
            extra.unlink()
    with pytest.raises(ValueError, match='independent groups'):
        dataset.prepare(multiclass_dataset, tmp_path / 's')


def test_invalid_trait_label_rejected(tmp_path):
    root = tmp_path / 'bad'
    root.mkdir()
    (root / 'labels.csv').write_text('image,' + ','.join(config.TRAITS) + '\nx.png,1,0,0.7,1,0\n')
    with pytest.raises(ValueError, match='0/1'):
        dataset.load_samples(root)


def test_hwxpi_conversion_maps_stability_to_neuroticism(tmp_path):
    src = tmp_path / 'hwxpi'
    (src / 'img').mkdir(parents=True)
    for sid in ('s1', 's2'):
        (src / 'img' / f'{sid}.jpg').write_bytes(b'x')
    data_file, solution_file = tmp_path / 'train.data', tmp_path / 'train.solution'
    data_file.write_text('s1 txt/s1.txt img/s1.jpg\ns2 txt/s2.txt img/s2.jpg\n')
    solution_file.write_text('1 0 1 1 0\n0 1 0 0 1\n')  # EXT AGR CON STA OPE
    labels = convert(data_file, solution_file, src, tmp_path / 'out')
    rows = list(csv.DictReader(labels.open()))
    assert [r['participant_id'] for r in rows] == ['s1', 's2']
    assert rows[0]['Extraversion'] == '1' and rows[0]['Neuroticism'] == '0'  # STA=1 -> low Neuroticism
    assert rows[1]['Openness'] == '1' and rows[1]['Neuroticism'] == '1'
