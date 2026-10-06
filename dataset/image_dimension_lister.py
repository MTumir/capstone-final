import argparse
from pathlib import Path

import numpy as np
import cv2 as cv

def list_image_dimensions(input, output):
    image_size_dict = {}

    # Prepare input directories.
    input_path = Path(f'{input}')
    input_list = list(input_path.glob('*/*'))    

    # For each image provided as input...
    for img_path in input_list:
        print(f'Opening image at {img_path}')

        # Grab image and log size.
        img = cv.imread(img_path)
        if type(img) is not np.ndarray:
            print(f'\t{img} not read as numpy.ndarray, skipping.')
            continue

        height, width, _ = img.shape
        img_dimension = f'{height}x{width}'
        if image_size_dict.get(img_dimension) is not None:
            image_size_dict[img_dimension] = image_size_dict[img_dimension] + 1
        else:
            image_size_dict[img_dimension] = 1

    # Print image_size_dict to output file.
    print(f'Image dimensions logged, writing to {output}')
    with open(output, "w") as file:
        file.write('*** Image Dimension Lister Output ***\n')
        file.write(f'Dictionary length: {len(image_size_dict)}\n\n')

        image_size_dict = dict(sorted(image_size_dict.items()))
        for entry in list(image_size_dict.items()):
            dimensions, occurrences = entry
            file.write(f'Dimension: {dimensions}\tOccurrences: {occurrences}\n')
    print(f'Writing to {output} complete!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='input/real/population_split/train', help='path to input directory.')
    parser.add_argument('-o', '--output', type=str, default='dataset/image_dimension_lister.out', help='path to output file.')
    args = parser.parse_args()

    print(f'Running image_dimension_lister.py with arguments:')
    print(f'\tInput = {args.input}')
    print(f'\tOutput = {args.output}')
    print()

    list_image_dimensions(args.input, args.output)
