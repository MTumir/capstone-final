import argparse
from pathlib import Path

import os
import pandas as pd

minAge = 5
maxAge = 70

def build_csv(input_dir, output_csv):
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
            listRows.append({
                "file_path": file_path,
                "age": age
            })

    df = pd.DataFrame(listRows)
    df.to_csv(output_csv, index=False)

    print(f'Saved CSV: {output_csv}')
    print(f'\nCSV Head:\n{df.head()}\n')
    print(f'Entries in CSV: {len(df)}')
    print(f'Minimum Age: {min(df["age"])}')
    print(f'Maximum Age: {max(df["age"])}')

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