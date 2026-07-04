"""Fine-tune a local Ultralytics YOLO model for this battery-board dataset.

Examples:
    # Recommended one-model baseline: boxes + class labels + masks
    python train_yolo.py --task segment --data battery_yolo_dataset/segment/data.yaml

    # Two-stage YOLO + SAM pipeline: train only the detector
    python train_yolo.py --task detect --data battery_yolo_dataset/detect/data.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("detect", "segment"), default="segment")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default=None, help="Override the pretrained checkpoint")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default=None, help="Examples: 0, cpu, mps")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--project", default="runs/battery")
    parser.add_argument("--name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_model = "yolo11n-seg.pt" if args.task == "segment" else "yolo11n.pt"
    model_path = args.model or default_model
    run_name = args.name or f"{args.task}_baseline"

    model = YOLO(model_path)
    train_kwargs = {
        "data": str(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": args.project,
        "name": run_name,
        "patience": 35,
        "pretrained": True,
        "plots": True,
        "seed": 42,
        # Conservative transforms for a fixed physical board. Increase later if needed.
        "degrees": 8.0,
        "translate": 0.08,
        "scale": 0.20,
        "fliplr": 0.0,
        "flipud": 0.0,
        "mosaic": 0.25,
        "close_mosaic": 15,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device

    model.train(**train_kwargs)
    model.val(data=str(args.data), split="test")


if __name__ == "__main__":
    main()
