"""load_json must read UTF-8 files the same way on every operating system (Windows defaults to cp1252)."""
import json

from src.services.prediction_service import load_json


def test_load_json_reads_utf8_text(tmp_path):
    path = tmp_path / 'metrics.json'
    path.write_bytes(json.dumps({'name': 'ResNet18 \u00b7 experimental \u00d7 2'}, ensure_ascii=False).encode('utf-8'))
    assert load_json(path)['name'] == 'ResNet18 \u00b7 experimental \u00d7 2'


def test_load_json_missing_file_returns_none(tmp_path):
    assert load_json(tmp_path / 'absent.json') is None


def test_load_json_invalid_json_returns_none(tmp_path):
    path = tmp_path / 'broken.json'
    path.write_text('{not json', encoding='utf-8')
    assert load_json(path) is None
