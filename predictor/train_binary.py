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

from dataset import AgeDataset
from model import build_model
from train_model import get_training_transforms

IMAGE_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 10
LR = 3e-4
MIN_AGE = 5
MAX_AGE = 70
AGE_RANGE = MAX_AGE - MIN_AGE
NUM_WORKERS = 0
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

    train_ds = AgeDataset(train_csv, transform=train_tf)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    classifier = build_model(effnet_index).to(DEVICE)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(classifier.parameters())

    ckpt_path = Path(output_pth)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    for epoch in range(EPOCHS):
        running_loss = 0.0
        num_inputs = 0
        for i, data in enumerate(train_loader):
            classifier.zero_grad()
            inputs = data[0].to(DEVICE)
            labels = data[1].type(torch.FloatTensor).to(DEVICE)
            output = classifier(inputs).view(-1)
            err = criterion(output, labels)
            running_loss += err * inputs.size(0)
            num_inputs += inputs.size(0)
            err.backward()
            optimizer.step()
        running_loss = running_loss / num_inputs
        print(epoch + 1, running_loss)

        # Save only when improved
        if running_loss < best_loss:
            best_loss = running_loss
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
            )
            print(f"saved new best to {ckpt_path} (loss={best_loss:.6f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train_csv', type=str, default='csv_builder_output/train.csv', help='path to training csv (from csv_builder.py).')
    parser.add_argument('-v', '--validation_csv', type=str, default='csv_builder_output/validation.csv', help='path to validation csv (from csv_builder.py).')
    parser.add_argument('-e', '--effnet_index', type=int, default=0, help='index of EfficientNet model.')
    parser.add_argument('-f', '--feature', type=int, default=0, help='feature to be extracted.')
    parser.add_argument('-s', '--seed', type=int, default=67, help='seed for PyTorch.')
    parser.add_argument('-o', '--output_pth', type=str, default='train_output/model.pth', help='path to output pth.')
    args = parser.parse_args()

    print(f'Running train_model.py with arguments:')
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