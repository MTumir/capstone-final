import argparse
from pathlib import Path

import numpy as np
import cv2 as cv

def yunet_filter(input, output, score_threshold, model_path):
    good_count = 0
    bad_count = 0

    # Prepare input directories.
    input_path = Path(f'{input}')
    fake_list = list(input_path.glob('*/*'))
    input_list = fake_list

    # Prepare output directories.
    output_path = Path(f'./{output}/')
    for i in range(0,101):
        output_dir = Path(f'{output_path}/good/{i}')
        if not output_dir.is_dir():
            output_dir.mkdir(parents=True)
        output_dir = Path(f'{output_path}/bad/{i}')
        if not output_dir.is_dir():
            output_dir.mkdir(parents=True)

    # Initialize YuNet detector.
    detector = cv.FaceDetectorYN.create(
        model_path,                           # model
        "",                                   # config
        (320, 320),                           # input_size
        score_threshold,                      # score_threshold
        0.3,                                  # nms_threshold
        5000                                  # top_k
    )

    # For each image provided as input...
    for img_path in input_list:
        print(f'Opening image at {img_path}')

        # Grab image and resize.
        img = cv.imread(img_path)
        if type(img) is not np.ndarray:
            print(f'\t{img} not read as numpy.ndarray, skipping.')
            continue

        img = cv.resize(img, (320, 320))
        width, height, _ = img.shape
        detector.setInputSize((height, width))

        # Run image through YuNet for face detection.
        face = detector.detect(img)

        # Determine image quality if face was detected.
        if face[1] is None:
            img_quality = 'bad'
            bad_count += 1
        else:
            img_quality = 'good'
            good_count += 1

        # Save image in appropriate directory.
        final_path = f'{output_path}/{img_quality}/{img_path.parent.name}/{img_path.name}'
        cv.imwrite(final_path, img)
        print(f'\t[{img_quality.upper()}] Saved {img_path.name} at {final_path}')

    # Print conclusion message.
    total_count = good_count + bad_count
    print('*** YuNet filtering complete! ***')
    print(f'\tImages Filtered: {total_count}')
    print(f'\tGood Image Count: {good_count} ({float(good_count) / float(total_count)})')
    print(f'\tBad Image Count: {bad_count} ({float(bad_count) / float(total_count)})')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='input/fake/stylegan', help='path to input directory.')
    parser.add_argument('-o', '--output', type=str, default='filtered_output/', help='path to output directory.')
    parser.add_argument('-s', '--score_threshold', type=float, default=0.85, help='yunet score threshold to pass for good sample.')
    parser.add_argument('-m', '--model_path', type=str, default='face_detection_yunet_2026may.onnx', help='path to YuNet onnx file.')
    args = parser.parse_args()

    print(f'Running yunet_filter.py with arguments:')
    print(f'\tInput = {args.input}')
    print(f'\tOutput = {args.output}')
    print(f'\tScore Threshold = {args.score_threshold}')
    print(f'\tModel Path = {args.model_path}')
    print()

    yunet_filter(args.input, args.output, args.score_threshold, args.model_path)