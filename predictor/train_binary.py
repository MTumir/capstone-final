import argparse
import random
from multiprocessing.spawn import freeze_support
from pathlib import Path

import cv2 as cv
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
from tqdm import tqdm
import dill

from dataset import AgeDataset
from model import build_model
from train_model import get_training_transforms, get_validation_transforms

IMAGE_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 10
MAXIMUM_BAD_EPOCHS = 3
LR = 3e-4
MIN_AGE = 5
MAX_AGE = 70
AGE_RANGE = MAX_AGE - MIN_AGE
NUM_WORKERS = 4
NUM_CHANNELS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True

YUNET_INPUT_SIZE = 320
YUNET_SCORE_THRESHOLD = 0.85

def train_binary(train_csv, validation_csv, effnet_index, feature, seed, output_pth):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

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

    classifier = build_model(effnet_index).to(DEVICE)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(classifier.parameters())

    ckpt_path = Path(output_pth)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        # Training step
        print('Training...')
        classifier.train()
        running_loss = 0.0
        num_images = 0
        for images, ages in tqdm(train_loader, desc='train'):
            images = images.to(DEVICE, non_blocking=True)
            ages = ages.to(DEVICE, non_blocking=True).unsqueeze(1)

            classifier.zero_grad()
            output = classifier(images).view(-1)
            err = criterion(output, ages)
            running_loss += err * images.size(0)
            num_images += images.size(0)
            err.backward()
            optimizer.step()
        train_loss = running_loss / num_images
        print(epoch + 1, train_loss)

        # Validation step
        print('Evaluating...')
        classifier.eval()
        running_loss = 0.0
        num_images = 0
        for images, ages in tqdm(validation_loader, desc='validation'):
            images = images.to(DEVICE, non_blocking=True)
            ages = ages.to(DEVICE, non_blocking=True).unsqueeze(1)

            classifier.zero_grad()
            output = classifier(images).view(-1)
            err = criterion(output, ages)
            running_loss += err * images.size(0)
            num_images += images.size(0)
            err.backward()
            optimizer.step()
        validation_loss = running_loss / num_images

        # Save only when improved
        if validation_loss < best_loss:
            best_loss = validation_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "model_state_dict": classifier.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "image_size": IMAGE_SIZE,
                    "num_channels": NUM_CHANNELS,
                },
                ckpt_path,
                pickle_module=dill
            )
            print(f"saved new best to {ckpt_path} (loss={best_loss:.6f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= MAXIMUM_BAD_EPOCHS:
                print(f'{epochs_without_improvement} epochs passed with no improvement. Exiting.')
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train_csv', type=str, default='csv_builder_output/train.csv', help='path to training csv (from csv_builder.py).')
    parser.add_argument('-v', '--validation_csv', type=str, default='csv_builder_output/validation.csv', help='path to validation csv (from csv_builder.py).')
    parser.add_argument('-e', '--effnet_index', type=int, default=0, help='index of EfficientNet model.')
    parser.add_argument('-f', '--feature', type=int, default=0, help='feature to be extracted.')
    parser.add_argument('-s', '--seed', type=int, default=67, help='seed for PyTorch.')
    parser.add_argument('-o', '--output_pth', type=str, default='train_output/model.pth', help='path to output pth.')
    args = parser.parse_args()

    print(f'Running train_binary.py with arguments:')
    print(f'\tTraining CSV = {args.train_csv}')
    print(f'\tValidation CSV = {args.validation_csv}')
    print(f'\tModel = EfficientNetB{args.effnet_index}')
    print(f'\tFeature = {args.feature}')
    print(f'\tSeed = {args.seed}')
    print(f'\tOutput PTH = {args.output_pth}')
    print()

    freeze_support()
    train_binary(args.train_csv, args.validation_csv, args.effnet_index,
                 args.feature, args.seed, args.output_pth)