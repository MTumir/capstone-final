import argparse

import os
from collections import defaultdict
import csv
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import AgeDataset
from model import build_model

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MIN_AGE = 5
MAX_AGE = 70
AGE_RANGE = MAX_AGE - MIN_AGE

IMAGE_SIZE = 128
BATCH_SIZE = 32

trans = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_model(model_p):
    save_model = torch.load(model_p, map_location=DEVICE)
    model = build_model().to(DEVICE)
    model.load_state_dict(save_model["model_state_dict"])
    model.eval()
    return model


def eval(csv_path, modelp, age_thresholds=(18,), output_dir="./eval_outputs"):
    os.makedirs(output_dir, exist_ok=True)

    evaluation_dataset = AgeDataset(csv_path, transform=trans)

    loader = DataLoader(
        evaluation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = load_model(modelp)

    errors = []
    errors_by_age = defaultdict(list)

    all_true_years = []
    all_pred_years = []

    with torch.no_grad():
        for images, ages in tqdm(loader, desc="evaluating"):
            images = images.to(DEVICE, non_blocking=True)
            ages = ages.to(DEVICE, non_blocking=True).unsqueeze(1)

            preds = model(images)

            pred_years = preds * AGE_RANGE + MIN_AGE
            true_years = ages * AGE_RANGE + MIN_AGE

            batch_errors = (pred_years - true_years).cpu().numpy().flatten()
            batch_true_years = true_years.cpu().numpy().flatten()
            batch_pred_years = pred_years.cpu().numpy().flatten()

            errors.extend(batch_errors)
            all_true_years.extend(batch_true_years.tolist())
            all_pred_years.extend(batch_pred_years.tolist())

            for true_age, pred_age, err in zip(batch_true_years, batch_pred_years, batch_errors):
                age_bucket = int(round(true_age))
                errors_by_age[age_bucket].append(err)

    errors = np.array(errors)
    all_true_years = np.array(all_true_years)
    all_pred_years = np.array(all_pred_years)

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    medae = np.median(np.abs(errors))
    bias = np.mean(errors)
    std_err = np.std(errors)
    max_err = np.max(np.abs(errors))

    print("\n=== Overall Regression Metrics ===")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"Median AE: {medae:.2f}")
    print(f"Bias: {bias:.2f}")
    print(f"Std Error: {std_err:.2f}")
    print(f"Max Error: {max_err:.2f}")
    print(f"Samples: {len(errors)}")

    per_age_reg_path = os.path.join(output_dir, "per_age_regression.csv")
    with open(per_age_reg_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["age", "count", "mae", "bias", "rmse"])

        for age in sorted(errors_by_age.keys()):
            age_errors = np.array(errors_by_age[age])
            writer.writerow([
                age,
                len(age_errors),
                np.mean(np.abs(age_errors)),
                np.mean(age_errors),
                np.sqrt(np.mean(age_errors ** 2))
            ])

    print(f"\nSaved: {per_age_reg_path}")

    summary_rows = []
    per_age_cls_rows = []

    for threshold in age_thresholds:
        true_cls = all_true_years >= threshold
        pred_cls = all_pred_years >= threshold

        tp = int(np.sum(pred_cls & true_cls))
        tn = int(np.sum((~pred_cls) & (~true_cls)))
        fp = int(np.sum(pred_cls & (~true_cls)))
        fn = int(np.sum((~pred_cls) & true_cls))

        total = len(all_true_years)
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
        for age in sorted(set(int(round(x)) for x in all_true_years)):
            mask = np.round(all_true_years).astype(int) == age
            if np.sum(mask) == 0:
                continue

            age_true = all_true_years[mask]
            age_pred = all_pred_years[mask]

            age_acc = np.mean((age_true >= threshold) == (age_pred >= threshold))

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


def singlepred(filepath, modelpath):
    model = load_model(modelpath)
    pred_img = Image.open(filepath).convert('RGB')
    pred_img = trans(pred_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(pred_img)

    predicted_age = pred.item() * AGE_RANGE + MIN_AGE
    print(f"Predicted Age: {round(predicted_age)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--input_csv', type=str, default='csv_builder_output/data.csv', help='path to input csv (from csv_builder.py).')
    parser.add_argument('-p', '--input_pth', type=str, default='train_output/model.pth', help='path to input pth (from train.py).')
    args = parser.parse_args()

    print(f'Running test.py with arguments:')
    print(f'\tInput CSV = {args.input_csv}')
    print(f'\tInput PTH = {args.input_pth}')
    print()

    eval(args.input_csv, args.input_pth, age_thresholds=(13, 15, 18, 21), output_dir="./eval_outputs_uniform70_stylegan")
