import argparse
import random
from pathlib import Path

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import torch
import shap
from PIL import Image

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../predictor/')))
from test_model import load_model, trans

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MIN_AGE = 5
MAX_AGE = 70
AGE_RANGE = MAX_AGE - MIN_AGE

IMAGE_SIZE = 128
BATCH_SIZE = 32

def shap_analysis(input, output, input_csv, input_pth, max_evals=500, batch_size=50):
    # Prepare output directories.
    output_dir = Path(output)
    if not output_dir.is_dir():
        output_dir.mkdir()
    for i in range(5,71):
        output_dir = Path(f'{output}/{i}')
        if not output_dir.is_dir():
            output_dir.mkdir()

    # Grab the original image for 975 faces, 15 from each real age.
    input_path = Path(f'./{input}/')
    input_list = list(input_path.glob('*'))
    sample_list = []
    for dir in input_list:
        files = [f for f in dir.iterdir() if f.is_file()]
        # for i in range(0,15):
        for i in range(0,1):
            image = random.choice(files)
            image_name = image.name
            age = image.parent.name
            image = cv.imread(image)
            image = cv.resize(image, (320, 320))
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            sample_list.append((image, image_name, age))

    # Prepare trained model.
    model = load_model(input_pth)
    if model is None:
        print(f'Model not loaded, exiting. [{input_pth}]')

    # Define confidence function for shap.Explainer.
    def __get_age_predictor_confidence(images):
        scores = []
        with torch.no_grad():
            for image in images:
                # Convert from cv2 to PIL image.
                image = Image.fromarray(image)
                image = trans(image).unsqueeze(0).to(DEVICE)

                prediction = model(image)
                # probabilities = torch.nn.functional.softmax(prediction, dim=1)
                # confidence, _ = torch.max(probabilities, dim=1)
                confidence, _ = torch.max(prediction, dim=1)
                scores.append(confidence)
        return np.array(scores)

    # Define masker and explainer for image.
    test_image, _, _ = sample_list[0]
    masker = shap.maskers.Image("blur(12,12)", test_image.shape)
    explainer = shap.Explainer(__get_age_predictor_confidence, masker)

    # Save plot of each image to disk.
    for sample in sample_list:
        image, image_name, age = sample
        shap_values = explainer(np.array([image]), max_evals=max_evals, batch_size=batch_size)

        shap.image_plot(shap_values, show=False)
        plt.savefig(f'{args.output}/{age}/{image_name}')
        plt.close()
        print(f'Saved plot of age {age} ({image_name})')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='input/', help='path to input directory.')
    parser.add_argument('-o', '--output', type=str, default='shap_age_predictor_output/', help='path to output directory.')
    parser.add_argument('-c', '--input_csv', type=str, default='csv_builder_output/data.csv', help='path to input csv (from csv_builder.py).')
    parser.add_argument('-p', '--input_pth', type=str, default='predictor/trained_models/styleganNormal80_18.pth', help='path to input pth (from train.py).')
    parser.add_argument('-e', '--max_evals', type=int, default=500, help='maximum number of evaluations allowed per image.')
    parser.add_argument('-b', '--batch_size', type=int, default=50, help='number of images inspected between model parameter updates.')
    args = parser.parse_args()

    print(f'Running shap_age_predictor.py with arguments:')
    print(f'\tInput = {args.input}')
    print(f'\tOutput = {args.output}')
    print(f'\tInput CSV = {args.input_csv}')
    print(f'\tInput PTH = {args.input_pth}')
    print(f'\tMax Evals = {args.max_evals}')
    print(f'\tBatch Size = {args.batch_size}')
    print()

    shap_analysis(args.input, args.output, args.input_csv, args.input_pth,
                  args.max_evals, args.batch_size)