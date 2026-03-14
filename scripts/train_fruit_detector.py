"""Train a YOLO fruit detector with ~10-50 images per fruit class.

Expected dataset layout:
  <dataset_root>/
    images/
      train/*.jpg
      val/*.jpg
    labels/
      train/*.txt
      val/*.txt

Label format per line (YOLO):
  <class_id> <x_center> <y_center> <width> <height>

Class names should be seedIds, e.g. tao, chuoi, xoai...
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import List

from ultralytics import YOLO

FRUIT_SEED_IDS: List[str] = [
    "tao",
    "dau_tay",
    "cam",
    "chanh",
    "sung",
    "dua",
    "chuoi",
    "mit",
    "na",
    "luu",
    "nho",
    "dua_hau",
    "du_du",
    "xoai",
    "bo",
    "vai",
    "chom_chom",
    "thanh_long",
    "kiwi",
    "chanh_dau",
    "dau_den",
    "dau_xanh",
    "phuc_bon_tu",
    "le",
    "dao",
    "man",
    "mo",
    "anh_dao",
    "oliu",
    "cha_la",
    "dua_xiem",
    "buoi",
]


def build_dataset_yaml(dataset_root: Path, output_yaml: Path, names: List[str]) -> None:
    content = {
        "path": str(dataset_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": names,
        "nc": len(names),
    }
    output_yaml.write_text(
        "\n".join(
            [
                f"path: {content['path']}",
                f"train: {content['train']}",
                f"val: {content['val']}",
                f"nc: {content['nc']}",
                "names:",
            ]
            + [f"  - {name}" for name in names]
        )
        + "\n",
        encoding="utf-8",
    )


def count_train_images(dataset_root: Path) -> int:
    train_dir = dataset_root / "images" / "train"
    if not train_dir.exists():
        return 0
    count = 0
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        count += len(list(train_dir.glob(ext)))
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fruit detector model")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--base-model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="runs")
    parser.add_argument("--name", default="fruit_detector")
    parser.add_argument("--output-model", type=Path, default=Path("models/fruit/best.pt"))
    args = parser.parse_args()

    dataset_root = args.dataset_root
    dataset_yaml = dataset_root / "fruit_dataset.yaml"
    build_dataset_yaml(dataset_root, dataset_yaml, FRUIT_SEED_IDS)

    train_image_count = count_train_images(dataset_root)
    if train_image_count < len(FRUIT_SEED_IDS) * 10:
        print(
            "[WARN] Training set looks small. Aim for ~10-50 images per fruit class "
            f"(recommended minimum total: {len(FRUIT_SEED_IDS) * 10})."
        )

    model = YOLO(args.base_model)
    model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        shear=2,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
    )

    best_from_runs = Path(args.project) / args.name / "weights" / "best.pt"
    if not best_from_runs.exists():
        raise FileNotFoundError(f"Trained weights not found: {best_from_runs}")

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_from_runs, args.output_model)

    metadata = {
        "dataset_root": str(dataset_root.resolve()),
        "classes": FRUIT_SEED_IDS,
        "epochs": args.epochs,
        "base_model": args.base_model,
        "weights": str(args.output_model.resolve()),
    }
    meta_path = args.output_model.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved trained model to: {args.output_model}")
    print(f"Saved metadata to: {meta_path}")


if __name__ == "__main__":
    main()
