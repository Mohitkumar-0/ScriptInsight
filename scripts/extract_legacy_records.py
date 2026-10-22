"""Extract the legacy ResNet50's recorded training log and test metrics from the notebook's saved outputs.

The values are copied verbatim from personality_prediction_resnet.ipynb outputs. They could not be
re-computed because the training dataset is not available. Output:
artifacts/legacy_resnet50/recorded_metrics.json

Usage:
    python scripts/extract_legacy_records.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

NOTEBOOK = config.BASE_DIR / 'personality_prediction_resnet.ipynb'
ANSI = re.compile(r'\x1b\[[0-9;]*m')
SUMMARY = re.compile(r'^\s*(\d+)/(\d+) .*val_accuracy')
METRIC = re.compile(r'([a-z_0-9]+): ([0-9.e+-]+)')


def cell_text(cell):
    return ANSI.sub('', ''.join(''.join(o.get('text', [])) for o in cell.get('outputs', [])))


def find_cell(cells, needle):
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'code' and needle in ''.join(cell['source']):
            return i, cell
    raise LookupError(needle)


def parse_history(text, phase):
    """The epoch number comes from the 'Epoch N: val_accuracy ...' callback line preceding each summary.
    Keras reports a full 'k/k' bar even when the generator ran out after one batch; such epochs are
    detected because their summary loss equals the loss printed after step 1."""
    epochs, current, first_step = {}, None, None
    for line in text.splitlines():
        if re.match(r'^Epoch \d+/\d+', line):
            first_step = None
            continue
        marker = re.match(r'^Epoch (\d+): ', line)
        if marker:
            current = int(marker.group(1))
            continue
        step = re.match(r'^\s*1/\d+ .*loss: ([0-9.]+)', line)
        if step and 'val_' not in line:
            first_step = float(step.group(1))
            continue
        summary = SUMMARY.match(line)
        if summary and current is not None and current not in epochs:
            values = {k: float(v) for k, v in METRIC.findall(line)}
            epochs[current] = {
                'phase': phase, 'epoch': current,
                'partial_epoch': first_step is not None and values.get('loss') == first_step,
                'train_loss': values.get('loss'), 'train_accuracy': values.get('accuracy'),
                'val_loss': values.get('val_loss'), 'val_accuracy': values.get('val_accuracy'),
                'learning_rate': values.get('learning_rate'),
            }
    return [epochs[k] for k in sorted(epochs)]


def parse_classification_report(text):
    per_class = {}
    for name in config.LEGACY_CLASSES:
        m = re.search(rf'{name}\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)', text)
        if m:
            per_class[name] = {'precision': float(m[1]), 'recall': float(m[2]), 'f1': float(m[3]), 'support': int(m[4])}
    return per_class


def main():
    cells = json.loads(NOTEBOOK.read_text())['cells']
    _, train_cell = find_cell(cells, 'history = model.fit(')
    _, tune_cell = find_cell(cells, 'history_finetune = model.fit(')
    _, eval_cell = find_cell(cells, 'best_model.evaluate(')
    _, report_cell = find_cell(cells, 'classification_report(')
    _, data_cell = find_cell(cells, 'Training Data Structure')

    data_text = cell_text(data_cell)
    train_part, test_part = data_text.split('Test Data Structure', 1)
    train_counts = {k: int(v) for k, v in re.findall(r'^(\w+): (\d+) images', train_part, flags=re.M)}
    test_counts = {k: int(v) for k, v in re.findall(r'^(\w+): (\d+) images', test_part, flags=re.M)}
    eval_text = cell_text(eval_cell)

    def grab(label):
        m = re.search(rf'{label}: ([0-9.]+)', eval_text)
        return float(m.group(1)) if m else None

    history = parse_history(cell_text(train_cell), 'frozen_backbone') + parse_history(cell_text(tune_cell), 'fine_tune')
    best_val = max((h['val_accuracy'] for h in history if h['val_accuracy'] is not None), default=None)

    record = {
        'provenance': ('Copied from the saved outputs of personality_prediction_resnet.ipynb (TensorFlow 2.20 / '
                       'Keras 3, CPU). Not re-computed: the training dataset is not available.'),
        'model': 'Legacy Keras ResNet50 + dense head (best_model.h5), single-label 5-category softmax',
        'dataset_recorded': {
            'source': "Local folder named 'archive' with 'augmented train' / 'augmented test' subfolders "
                      '(not in the repository; origin and label provenance undocumented).',
            'train_folder_counts': train_counts,
            'test_folder_counts': test_counts,
            'split': 'train folder split 80/20 into train/validation (random, stratified); separate test folder',
            'sizes': {'train': 2732, 'validation': 684, 'test': 660},
        },
        'test_metrics_recorded': {
            'accuracy': grab('Accuracy'),
            'weighted_precision': grab(r'Precision \(weighted\)'),
            'weighted_recall': grab(r'Recall \(weighted\)'),
            'weighted_f1': grab(r'F1-Score \(weighted\)'),
            'per_class': parse_classification_report(cell_text(report_cell)),
            'label_order': config.LEGACY_CLASSES,
            'confusion_matrix': None,
            'note': 'The notebook only plotted the confusion matrix as an image; numeric values were not saved.',
        },
        'best_validation_accuracy_recorded': best_val,
        'history': history,
        'history_note': ('Epochs with partial_epoch=true ran only 1 training batch because the data generator '
                         'ran out (steps_per_epoch mismatch); Keras still printed a full progress bar.'),
        'known_issues': [
            'Each training-class count is a multiple of 8, suggesting 8 augmented copies per original image; the '
            'random 80/20 split can place copies of one image in both train and validation (leakage risk).',
            'The saved checkpoint comes from the frozen-backbone phase; fine-tuning never improved validation '
            'accuracy, so it was not saved.',
            'Horizontal flips were used as augmentation, which mirrors handwriting.',
        ],
    }
    out = config.LEGACY_RECORDS_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    print(f'Wrote {out}: {len(history)} epochs, best val acc {best_val}, test acc {record["test_metrics_recorded"]["accuracy"]}')


if __name__ == '__main__':
    main()
