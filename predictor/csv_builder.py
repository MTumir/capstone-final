import argparse
from pathlib import Path

import os
import cv2 as cv
import pandas as pd

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../face_region_extractor/')))
from runtime_facial_region_extractor import FacialRegionExtractor, process_image, adjust_coordinates

minAge = 5
maxAge = 70

YUNET_INPUT_SIZE = 320
YUNET_SCORE_THRESHOLD = 0.85

def build_csv(input_dir, output_csv):
    fre = FacialRegionExtractor()
    good_samples = 0
    bad_samples = 0

    listRows = []
    output_dir = Path(output_csv).parent
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)

    for folder in os.listdir(input_dir):
        path = os.path.join(input_dir, folder)
        age = int(folder)
        files = []

        for file in os.listdir(path):
            files.append(os.path.join(path, file))

        for file_path in files:
            if fre.process_image_from_path(0, file_path) is not None:
                good_samples += 1
                listRows.append({
                    "file_path": file_path,
                    "age": age
                })
            else: bad_samples += 1

    df = pd.DataFrame(listRows)
    df.to_csv(output_csv, index=False)

    print(f'Saved CSV: {output_csv}')
    print(f'\nCSV Head:\n{df.head()}\n')
    print(f'Entries in CSV: {len(df)}')
    print(f'Minimum Age: {min(df["age"])}')
    print(f'Maximum Age: {max(df["age"])}')
    print(f'Good Samples: {good_samples}')
    print(f'Bad Samples: {bad_samples}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, default='balancing_output/', help='path to input dir (from balance_app.py).')
    parser.add_argument('-o', '--output_csv', type=str, default='csv_builder_output/data.csv', help='path to output csv.')
    args = parser.parse_args()

    print(f'Running csv_builder.py with arguments:')
    print(f'\tInput Dir = {args.input_dir}')
    print(f'\tOutput CSV = {args.output_csv}')
    print()

    build_csv(args.input_dir, args.output_csv)