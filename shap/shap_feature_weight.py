import argparse
import random
from pathlib import Path

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import shap

def get_yunet_confidence(images):
    scores = []
    for image in images:
        _, face = detector.detect(image)
        if face is None: scores.append(0.0)
        else: scores.append(face[0][14])
    return np.array(scores)

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, default='input', help='path to input folder.')
parser.add_argument('-o', '--output', type=str, default='output', help='path to output folder. will be filled with identical folder structure as input.')
parser.add_argument('-m', '--model_path', type=str, default='face_detection_yunet_2026may.onnx', help='path to YuNet onnx file.')
parser.add_argument('-e', '--max_evals', type=int, default=500, help='maximum number of evaluations allowed per image.')
parser.add_argument('-b', '--batch_size', type=int, default=50, help='number of images inspected between model parameter updates.')
args = parser.parse_args()

print(f'Running {Path(__file__).name} with arguments:')
print(f'\tInput = {args.input}')
print(f'\tOutput = {args.output}')
print(f'\tModel Path = {args.model_path}')
print(f'\tMax Evals = {args.max_evals}')
print(f'\tBatch Size = {args.batch_size}')
print()

# Prepare output directories.
output_dir = Path(args.output)
if not output_dir.is_dir():
    output_dir.mkdir()
for i in range(5,71):
    output_dir = Path(f'{args.output}/{i}')
    if not output_dir.is_dir():
        output_dir.mkdir()

# Grab the original image for 975 faces, 15 from each real age.
input_path = Path(f'./{args.input}/')
real_list = list(input_path.glob('real/population_split/train/*'))
sample_list = []
for dir in real_list:
    files = [f for f in dir.iterdir() if f.is_file()]
    for i in range(0,15):
        image = random.choice(files)
        image_name = image.name
        age = image.parent.name
        image = cv.imread(image)
        image = cv.resize(image, (320, 320))
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        sample_list.append((image, image_name, age))

# Define masker and explainer for image.
test_image = cv.imread(f'{args.input}/real/population_split/train/5/0.jpg')
test_image = cv.resize(test_image, (320, 320))
test_image = cv.cvtColor(test_image, cv.COLOR_BGR2RGB)
masker = shap.maskers.Image("blur(12,12)", test_image.shape)
explainer = shap.Explainer(get_yunet_confidence, masker)

# Initialize YuNet detector.
detector = cv.FaceDetectorYN.create(
    args.model_path,     # model
    "",                  # config
    (320, 320),          # input_size
    0.85,                # score_threshold
    0.3,                 # nms_threshold
    5000                 # top_k
)

# Save plot of each image to disk.
for sample in sample_list:
    image, image_name, age = sample
    shap_values = explainer(np.array([image]), max_evals=args.max_evals, batch_size=args.batch_size)
    shap.image_plot(shap_values, show=False)
    plt.savefig(f'{args.output}/{age}/{image_name}')
    plt.close()
    print(f'Saved plot of age {age} ({image_name})')
