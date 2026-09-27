from __future__ import annotations

import pickle
import tkinter as tk
from tkinter import ttk, filedialog

import math
import random
import shutil
from pathlib import Path
from typing import Iterable, Sequence
import cv2 as cv

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../face_region_extractor/')))
from runtime_facial_region_extractor import *

def create_train_test_split(
        src_root: str | Path,
        train_root: str | Path,
        test_root: str | Path,
        distribution: str = "uniform",
        population_percentages: Sequence[float] | None = None,
        seed: int | None = 42,
        copy_files: bool = True,
        image_extensions: Iterable[str] = (".png", ".jpg", ".jpeg", ".bmp", ".webp"),
) -> None:
    src_root = Path(src_root)
    train_root = Path(train_root)
    test_root = Path(test_root)

    rng = random.Random(seed)
    valid_exts = {ext.lower() for ext in image_extensions}

    age_to_files: dict[int, list[Path]] = {}
    for age in range(101):
        age_dir = src_root / str(age)
        if not age_dir.exists():
            age_to_files[age] = []
            continue
        age_to_files[age] = [
            p for p in age_dir.iterdir()
            if p.is_file() and p.suffix.lower() in valid_exts
        ]

    counts = {age: len(files) for age, files in age_to_files.items()}

    target_counts: list[int] = []

    if distribution == "uniform":
        for age in range(101):
            target_counts.append(76)

    else:
        s = sum(population_percentages[5:71])
        total_test_size = math.ceil(5000 / s)
        target_counts = [math.ceil(p * total_test_size) for p in population_percentages]

    # Final test counts with fallback for underrepresented age groups:
    # if available < target, use 50% split for that age
    test_counts: list[int] = []
    for age in range(101):
        if counts[age] < target_counts[age]:
            test_counts.append(math.floor(0.5 * counts[age]))
        else:
            test_counts.append(target_counts[age])

    # Create output directories
    train_root.mkdir(parents=True, exist_ok=True)
    test_root.mkdir(parents=True, exist_ok=True)

    total_train = 0
    total_test = 0

    for age in range(5, 71):
        files = list(age_to_files[age])
        rng.shuffle(files)

        n_test = test_counts[age]
        test_files = files[:n_test]
        train_files = files[n_test:]

        train_age_dir = train_root / str(age)
        test_age_dir = test_root / str(age)
        train_age_dir.mkdir(parents=True, exist_ok=True)
        test_age_dir.mkdir(parents=True, exist_ok=True)

        for src_path in test_files:
            dst_path = test_age_dir / src_path.name
            if copy_files:
                shutil.copy2(src_path, dst_path)
            else:
                shutil.move(src_path, dst_path)

        for src_path in train_files:
            dst_path = train_age_dir / src_path.name
            if copy_files:
                shutil.copy2(src_path, dst_path)
            else:
                shutil.move(src_path, dst_path)

        total_test += len(test_files)
        total_train += len(train_files)

    print(f"Created split using distribution='{distribution}'")
    print(f"Total train images: {total_train}")
    print(f"Total test images:  {total_test}")

    print("\nPer-age summary:")
    print("age | available | target | actual_test | actual_train")
    print("-" * 55)
    for age in range(101):
        available = counts[age]
        target = target_counts[age]
        actual_test = test_counts[age]
        actual_train = available - actual_test
        print(f"{age:3d} | {available:9d} | {target:6d} | {actual_test:11d} | {actual_train:12d}")


def balance(dist, percent_real, src1, src2, dst, age_min=5, age_max=70):
    percent_real = float(percent_real) / 100
    if dist == "US Population":
        dist = pickle.load(open("stats.pkl", "rb"))['us24']
    elif dist == "Centered Normal (mean 13)":
        dist = pickle.load(open("stats.pkl", "rb"))['13']
    elif dist == "Centered Normal (mean 15)":
        dist = pickle.load(open("stats.pkl", "rb"))['15']
    elif dist == "Centered Normal (mean 18)":
        dist = pickle.load(open("stats.pkl", "rb"))['18']
    elif dist == "Centered Normal (mean 21)":
        dist = pickle.load(open("stats.pkl", "rb"))['21']

    src1 = Path(src1)
    src2 = Path(src2)
    dst = Path(dst)

    fre = FacialRegionExtractor(model_path='../../face_detection_yunet_2026may.onnx')

    ages = list(range(age_min, age_max + 1))
    rng = random.Random(42)
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    def list_images(root: Path, age: int) -> list[Path]:
        age_dir = root / str(age)
        if not age_dir.exists():
            return []
        return sorted(
            p for p in age_dir.iterdir()
            if p.is_file() and p.suffix.lower() in valid_exts
        )

    def normalized_probs(dist, ages, age_min, age_max):
        if isinstance(dist, str):
            return [1.0 / len(ages)] * len(ages)
        vals = list(dist)
        probs = vals[age_min:age_max + 1]
        s = sum(probs)
        return [p / s for p in probs]

    def largest_remainder_allocation(total_size: int, probs: list[float], capacities: list[int]) -> list[int] | None:
        if total_size < 0 or total_size > sum(capacities):
            return None

        raw = [p * total_size for p in probs]
        alloc = [math.floor(x) for x in raw]

        for a, cap in zip(alloc, capacities):
            if a > cap:
                return None

        leftover = total_size - sum(alloc)
        remainders = sorted(
            [(raw[i] - alloc[i], i) for i in range(len(probs))],
            reverse=True,
        )

        for _, idx in remainders:
            if leftover == 0:
                break
            if alloc[idx] < capacities[idx]:
                alloc[idx] += 1
                leftover -= 1

        if leftover != 0:
            return None

        return alloc

    real_files = {age: list_images(src1, age) for age in ages}
    fake_files = {age: list_images(src2, age) for age in ages}

    real_counts = {age: len(real_files[age]) for age in ages}
    fake_counts = {age: len(fake_files[age]) for age in ages}
    total_counts = {age: real_counts[age] + fake_counts[age] for age in ages}

    total_available = sum(total_counts.values())

    probs = normalized_probs(dist, ages, age_min, age_max)
    capacities = [total_counts[age] for age in ages]

    best_result = None

    for total_size in range(1, total_available + 1):
        alloc = largest_remainder_allocation(total_size, probs, capacities)
        if alloc is None:
            continue

        bucket_sizes = {age: alloc[i] for i, age in enumerate(ages)}

        real_used = {
            age: min(real_counts[age], bucket_sizes[age])
            for age in ages
        }
        fake_used = {
            age: bucket_sizes[age] - real_used[age]
            for age in ages
        }

        if any(fake_used[age] > fake_counts[age] for age in ages):
            continue

        total_real_used = sum(real_used.values())
        actual_real_fraction = total_real_used / total_size

        error = abs(actual_real_fraction - percent_real)

        candidate = {
            "total_size": total_size,
            "bucket_sizes": bucket_sizes,
            "real_used": real_used,
            "fake_used": fake_used,
            "actual_real_fraction": actual_real_fraction,
            "error": error,
        }

        if best_result is None:
            best_result = candidate
        else:
            if candidate["error"] < best_result["error"]:
                best_result = candidate
            elif (
                    math.isclose(candidate["error"], best_result["error"], rel_tol=0.0, abs_tol=1e-12)
                    and candidate["total_size"] > best_result["total_size"]
            ):
                best_result = candidate

    dst.mkdir(parents=True, exist_ok=True)

    for age in ages:
        n_real = best_result["real_used"][age]
        n_fake = best_result["fake_used"][age]

        age_dir = dst / str(age)
        age_dir.mkdir(parents=True, exist_ok=True)

        real_pool = list(real_files[age])
        fake_pool = list(fake_files[age])

        rng.shuffle(real_pool)
        rng.shuffle(fake_pool)

        selected_real = real_pool[:n_real]
        selected_fake = fake_pool[:n_fake]

        for src_path in selected_real:
            img = fre.process_image(0, str(src_path))
            dst_path = age_dir / src_path.name
            if img is not None: cv.imwrite(dst_path, img)
            # shutil.copy2(src_path, dst_path)

        for src_path in selected_fake:
            img = fre.process_image(0, str(src_path))
            dst_path = age_dir / src_path.name
            if img is not None: cv.imwrite(dst_path, img)
            # shutil.copy2(src_path, dst_path)

    total_real_used = sum(best_result["real_used"].values())
    total_fake_used = sum(best_result["fake_used"].values())
    total_used = best_result["total_size"]

    print("Balanced dataset created.")
    print(f"Ages included:            {age_min}..{age_max}")
    print(f"Total images:             {total_used}")
    print(f"Real used:                {total_real_used}")
    print(f"Fake used:                {total_fake_used}")
    print(f"Requested real fraction:  {percent_real:.4f}")
    print(f"Actual real fraction:     {best_result['actual_real_fraction']:.4f}")


def main():
    root = tk.Tk()
    root.title("Dataset Balancer")
    root.resizable(True, True)

    tk.Label(
        root,
        text="Enter desired parameters",
        font=("Font", 24),
    ).pack(ipady=5, fill="x")

    def process_user_inputs():
        dist = dist_combobox.get()
        percent_real = percent_real_var.get()
        balance(dist, percent_real, src1, src2, dst)

    tk.Label(root, text="Distribution").pack(anchor="w", padx=30)
    dist_combobox = ttk.Combobox(root, values=["Uniform", "US Population", "Centered Normal (mean 13)", "Centered Normal (mean 15)", "Centered Normal (mean 18)", "Centered Normal (mean 21)"])
    dist_combobox.set("Uniform")
    dist_combobox.pack(padx=30, fill="x")

    tk.Label(root, text="% Real").pack(anchor="w", padx=30)
    percent_real_var = tk.StringVar(value="100")
    spinbox = tk.Spinbox(
        root,
        from_=0,
        to=100,
        textvariable=percent_real_var,
    )
    spinbox.pack(padx=30, fill="x")

    tk.Label(root, text="Real Source").pack(anchor="w", padx=30)
    real_source_entry = tk.Label(root)
    real_source_entry.pack(padx=30, fill="x")
    src1 = ""

    def get_real_dir():
        nonlocal src1
        src1 = filedialog.askdirectory(
            parent=root,
        )
        real_source_entry.config({'text': src1})

    tk.Button(
        root,
        text="Choose folder",
        command=get_real_dir,
        width=18,
        cursor="hand2",
    ).pack(pady=10, padx=30, fill="x")

    tk.Label(root, text="Fake Source").pack(anchor="w", padx=30)
    fake_source_entry = tk.Label(root)
    fake_source_entry.pack(padx=30, fill="x")
    src2 = ""

    def get_fake_dir():
        nonlocal src2
        src2 = filedialog.askdirectory(
            parent=root,
        )
        fake_source_entry.config({'text': src2})

    tk.Button(
        root,
        text="Choose folder",
        command=get_fake_dir,
        width=18,
        cursor="hand2",
    ).pack(pady=10, padx=30, fill="x")

    tk.Label(root, text="Output Location").pack(anchor="w", padx=30)
    dst_entry = tk.Label(root)
    dst_entry.pack(padx=30, fill="x")
    dst = ""

    def get_dst_dir():
        nonlocal dst
        dst = filedialog.askdirectory(
            parent=root,
        )
        dst_entry.config({'text': dst})

    tk.Button(
        root,
        text="Choose folder",
        command=get_dst_dir,
        width=18,
        cursor="hand2",
    ).pack(pady=10, padx=30, fill="x")

    # Sign in button
    tk.Button(
        root,
        text="Generate balanced dataset",
        command=process_user_inputs,
        width=18,
        cursor="hand2",
    ).pack(pady=10, padx=30, fill="x")

    root.mainloop()


if __name__ == "__main__":
    main()
