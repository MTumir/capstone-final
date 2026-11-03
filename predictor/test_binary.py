import argparse
import os
import csv

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import AgeDataset
from model import build_model
from test_model import get_test_transforms, load_model

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MIN_AGE = 5
MAX_AGE = 70
AGE_RANGE = MAX_AGE - MIN_AGE
IMAGE_SIZE = 128
BATCH_SIZE = 32

YUNET_INPUT_SIZE = 320
YUNET_SCORE_THRESHOLD = 0.85

# def load_model(model_p):
#     save_model = torch.load(model_p, map_location=DEVICE)
#     model = build_model().to(DEVICE)
#     model.load_state_dict(save_model["model_state_dict"])
#     model.eval()
#     return model


def eval(input_csv, input_pth, effnet_index, feature, threshold=18, output_dir="./eval_outputs"):
    os.makedirs(output_dir, exist_ok=True)

    yunet = cv.FaceDetectorYN.create(
        'face_detection_yunet_2026may.onnx',  # model
        "",                                   # config
        (YUNET_INPUT_SIZE, YUNET_INPUT_SIZE), # input_size
        YUNET_SCORE_THRESHOLD,                # score_threshold
        0.3,                                  # nms_threshold
        5000                                  # top_k
    )

    test_tf = get_test_transforms(yunet, feature)
    evaluation_dataset = AgeDataset(input_csv, transform=test_tf)

    loader = DataLoader(
        evaluation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = load_model(input_pth, effnet_index)

    actual = []
    predicted = []

    with torch.no_grad():
        for images, ages in tqdm(loader, desc="evaluating"):
            images = images.to(DEVICE, non_blocking=True)
            ages = ages.to(DEVICE, non_blocking=True).unsqueeze(1)
            preds = model(images).cpu().numpy().flatten().tolist()
            act = (ages * AGE_RANGE + MIN_AGE).cpu().numpy().flatten()
            actual.extend(act.tolist())
            predicted.extend(preds)
    actual = np.array(actual)
    predicted = np.array(predicted)

    summary_rows = []
    per_age_cls_rows = []

    true_cls = actual >= threshold
    pred_cls = predicted >= 0.5

    tp = int(np.sum(pred_cls & true_cls))
    tn = int(np.sum((~pred_cls) & (~true_cls)))
    fp = int(np.sum(pred_cls & (~true_cls)))
    fn = int(np.sum((~pred_cls) & true_cls))

    total = len(actual)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n=== Classification @ {threshold} ===")
    print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    summary_rows.append([
        threshold, accuracy, precision, recall, specificity, f1, tp, tn, fp, fn
    ])

    # per-age classification
    for age in sorted(set(int(round(x)) for x in actual)):
        mask = np.round(actual).astype(int) == age
        if np.sum(mask) == 0:
            continue

        age_true = actual[mask]
        age_pred = predicted[mask]

        age_acc = np.mean((age_true >= threshold) == (age_pred >= 0.5))

        per_age_cls_rows.append([
            threshold, age, np.sum(mask), age_acc
        ])

    summary_path = os.path.join(output_dir, "summary_metrics.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "threshold", "accuracy", "precision", "recall",
            "specificity", "f1", "tp", "tn", "fp", "fn"
        ])
        writer.writerows(summary_rows)

    print(f"Saved: {summary_path}")

    per_age_cls_path = os.path.join(output_dir, "per_age_classification.csv")
    with open(per_age_cls_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "age", "count", "accuracy"])
        writer.writerows(per_age_cls_rows)

    print(f"Saved: {per_age_cls_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--input_csv', type=str, default='csv_builder_output/data.csv', help='path to input csv (from csv_builder.py).')
    parser.add_argument('-p', '--input_pth', type=str, default='train_output/model.pth', help='path to input pth (from train.py).')
    parser.add_argument('-e', '--effnet_index', type=int, default=0, help='index of EfficientNet model.')
    parser.add_argument('-f', '--feature', type=int, default=0, help='feature to be extracted.')
    parser.add_argument('-o', '--output_dir', type=str, default='test_output/', help='directory for output.')
    args = parser.parse_args()

    print(f'Running test_binary.py with arguments:')
    print(f'\tInput CSV = {args.input_csv}')
    print(f'\tInput PTH = {args.input_pth}')
    print(f'\tModel = EfficientNetB{args.effnet_index}')
    print(f'\tFeature = {args.feature}')
    print(f'\tOutput Dir = {args.output_dir}')
    print()

    eval(args.input_csv, args.input_pth, args.effnet_index, args.feature,
        threshold=18, output_dir=args.output_dir)
