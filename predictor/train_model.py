import argparse

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2 as cv

from dataset import AgeDataset
from model import build_model

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../face_region_extractor/')))
from runtime_facial_region_extractor import FacialRegionWrapper

IMAGE_SIZE = 32
BATCH_SIZE = 64
EPOCHS = 10
LR = 3e-4
MIN_AGE = 5
MAX_AGE = 70
AGE_RANGE = MAX_AGE - MIN_AGE
NUM_WORKERS = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True

YUNET_INPUT_SIZE = 320
YUNET_SCORE_THRESHOLD = 0.85

def get_training_transforms(yunet, feature):
    return transforms.Compose([
        FacialRegionWrapper(yunet, feature),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def get_validation_transforms(yunet, feature):
    return transforms.Compose([
        FacialRegionWrapper(yunet, feature),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        # FacialRegionExtractor(feature),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(10),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def run(train_csv='csv_builder_output/train.csv', validation_csv='csv_builder_output/validation.csv', 
        save_path='train_output/model.pth', effnet_index=0, feature=0):
    print("Device:", DEVICE)

    yunet = cv.FaceDetectorYN.create(
        'face_detection_yunet_2026may.onnx',  # model
        "",                                   # config
        (YUNET_INPUT_SIZE, YUNET_INPUT_SIZE), # input_size
        YUNET_SCORE_THRESHOLD,                # score_threshold
        0.3,                                  # nms_threshold
        5000                                  # top_k
    )

    train_tf = get_training_transforms(yunet, feature)
    validation_tf = get_validation_transforms(yunet, feature)

    train_ds = AgeDataset(train_csv, transform=train_tf)
    validation_ds = AgeDataset(validation_csv, transform=validation_tf)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    validation_loader = DataLoader(
        validation_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    model = build_model(effnet_index).to(DEVICE)

    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    use_amp = (DEVICE == "cuda")

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        print('Training...')
        model.train()
        running_loss = 0.0

        for images, ages in tqdm(train_loader, desc="train"):
            images = images.to(DEVICE, non_blocking=True)
            ages = ages.to(DEVICE, non_blocking=True).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                preds = model(images)
                loss = criterion(preds, ages)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        train_loss = running_loss / max(1, len(train_loader))
        print(f"Train loss (normalized MAE): {train_loss:.4f}")

        scheduler.step()
        print("LR:", scheduler.get_last_lr()[0])

        print('Evaluating...')
        model.eval()
        running_loss = 0.0

        for images, ages in tqdm(validation_loader, desc="validation"):
            images = images.to(DEVICE, non_blocking=True)
            ages = ages.to(DEVICE, non_blocking=True).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                preds = model(images)
                loss = criterion(preds, ages)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        validation_loss = running_loss / max(1, len(validation_loader))
        print(f"Validation loss (normalized MAE): {validation_loss:.4f}")

        scheduler.step()
        print("LR:", scheduler.get_last_lr()[0])

    torch.save({
    "model_state_dict": model.state_dict(),
    "image_size": IMAGE_SIZE,
    "max_age": MAX_AGE,
    "min_age": MIN_AGE,},
    save_path)
    print(f"Saved best checkpoint -> {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train_csv', type=str, default='csv_builder_output/train.csv', help='path to training csv (from csv_builder.py).')
    parser.add_argument('-v', '--validation_csv', type=str, default='csv_builder_output/validation.csv', help='path to validation csv (from csv_builder.py).')
    parser.add_argument('-e', '--effnet_index', type=int, default=0, help='index of EfficientNet model.')
    parser.add_argument('-f', '--feature', type=int, default=0, help='feature to be extracted.')
    parser.add_argument('-o', '--output_pth', type=str, default='train_output/model.pth', help='path to output pth.')
    args = parser.parse_args()

    print(f'Running train_model.py with arguments:')
    print(f'\tTraining CSV = {args.train_csv}')
    print(f'\tValidation CSV = {args.validation_csv}')
    print(f'\tModel = EfficientNetB{args.effnet_index}')
    print(f'\tFeature = {args.feature}')
    print(f'\tOutput PTH = {args.output_pth}')
    print()

    run(args.train_csv, args.validation_csv,
        args.output_pth, args.effnet_index, args.feature)