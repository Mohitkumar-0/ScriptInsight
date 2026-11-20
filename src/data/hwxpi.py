"""Convert the ChaLearn HWxPI dataset into ScriptInsight's ``labels.csv`` layout.

HWxPI (ICPR 2018 challenge) provides, per split, a ``.data`` file with columns
``subject_id, transcript_path, image_path`` and a ``.solution`` file with five binary labels in the
order ``EXT AGR CON STA OPE`` (1 = high pole, 0 = low pole, assigned from TIPI questionnaire norms).
STA is Emotional Stability, the opposite pole of Neuroticism, so Neuroticism = 1 - STA.

Obtain the data yourself from https://chalearnlap.cvc.uab.cat/dataset/29/description/ (registration
on the CodaLab competition is required) and respect its terms of use.

Usage:
    python -m src.data.hwxpi --data-file train.data --solution-file train.solution \
        --image-root /path/to/hwxpi --out data/hwxpi
"""
import argparse
import csv
import re
import shutil
from pathlib import Path

import config

SOLUTION_ORDER = ['EXT', 'AGR', 'CON', 'STA', 'OPE']


def _split_row(line):
    return [p for p in re.split(r'[\s,;]+', line.strip()) if p]


def convert(data_file, solution_file, image_root, out_dir, copy_images=True):
    data_rows = [_split_row(l) for l in Path(data_file).read_text(encoding='utf-8-sig').splitlines() if l.strip()]
    sol_rows = [_split_row(l) for l in Path(solution_file).read_text(encoding='utf-8-sig').splitlines() if l.strip()]
    if len(data_rows) != len(sol_rows):
        raise ValueError(f'{data_file} has {len(data_rows)} rows but {solution_file} has {len(sol_rows)}.')

    out = Path(out_dir)
    (out / 'images').mkdir(parents=True, exist_ok=True)
    with (out / 'labels.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'participant_id', *config.TRAITS])
        for data, sol in zip(data_rows, sol_rows):
            if len(data) < 3 or len(sol) != 5 or any(v not in {'0', '1'} for v in sol):
                raise ValueError(f'Unexpected HWxPI row: data={data} solution={sol}')
            subject, image_rel = data[0], data[2]
            raw = dict(zip(SOLUTION_ORDER, map(int, sol)))
            trait = {'Openness': raw['OPE'], 'Conscientiousness': raw['CON'], 'Extraversion': raw['EXT'],
                     'Agreeableness': raw['AGR'], 'Neuroticism': 1 - raw['STA']}
            source = Path(image_root) / image_rel
            if not source.exists():
                raise FileNotFoundError(f'Image listed in {data_file} not found: {source}')
            target = Path('images') / f'{subject}_{source.name}'
            if copy_images:
                shutil.copy2(source, out / target)
            writer.writerow([str(target), subject, *[trait[t] for t in config.TRAITS]])
    return out / 'labels.csv'


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--data-file', required=True)
    parser.add_argument('--solution-file', required=True)
    parser.add_argument('--image-root', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(f'Wrote {convert(args.data_file, args.solution_file, args.image_root, args.out)}')


if __name__ == '__main__':
    main()
