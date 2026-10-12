import argparse
import shutil
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image

from model import Classifier

THRESHOLD = 0.98
IMAGE_SIZE = 64
NUM_CHANNELS = 3
NGPU = 1


def is_image(path: Path):
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def copy_preserve_structure(base_path: Path, src_path: Path, dst_root: Path):
    rel = src_path.relative_to(base_path)
    dst = dst_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst)


def classify_images(input_dir: Path, output_dir: Path, model_dir: str, subdir0: str, subdir1: str):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(Path(model_dir), map_location=device)

    model = Classifier(IMAGE_SIZE, NUM_CHANNELS, NGPU).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    image_paths = [p for p in input_dir.rglob("*") if p.is_file() and is_image(p)]

    with torch.no_grad():
        for idx, img_path in enumerate(image_paths):
            try:
                img = Image.open(img_path).convert("RGB")
                tensor = transform(img).unsqueeze(0).to(device)

                prob = model(tensor).view(-1).item()
                pred = 1 if prob >= THRESHOLD else 0

                if pred == 1:
                    copy_preserve_structure(input_dir, img_path, output_dir / subdir1)
                else:
                    copy_preserve_structure(input_dir, img_path, output_dir / subdir0)

                if idx % 500 == 0 and idx > 0:
                    print(f"Processed {idx}/{len(image_paths)}")

            except Exception as e:
                print(f"Skipped {img_path}: {e}")

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, default='input/', help='path to input directory.')
    parser.add_argument('-o', '--output_dir', type=str, default='classify_out/', help='path to output directory.')
    parser.add_argument('-m', '--model_path', type=str, default='dataset/filtering/trained_models/stage3.pt', help='path to input pth.')
    args = parser.parse_args()

    print(f'Running classify.py with arguments:')
    print(f'\tInput = {args.input_dir}')
    print(f'\tOutput = {args.output_dir}')
    print(f'\tModel = {args.model_path}')
    print()

    classify_images(Path(args.input_dir), Path(args.output_dir), args.model_path,
                    "rejected", "accepted")
