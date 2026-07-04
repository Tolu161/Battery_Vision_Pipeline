"""Prepare the uploaded Roboflow COCO-segmentation export for local Ultralytics training.

Creates matching YOLO detection and YOLO instance-segmentation datasets using the
same train/validation/test split.

Usage:
    python prepare_dataset.py \
        --zip "Battery cell case object detecti 2.coco-segmentation.zip" \
        --output battery_yolo_dataset

Dependencies:
    pip install opencv-python numpy pycocotools pyyaml
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

try:
    from pycocotools import mask as mask_utils
except ImportError as exc:
    raise SystemExit(
        "pycocotools is required. Install it with: pip install pycocotools"
    ) from exc



CLASS_RENAMES = {"empy-case-tray": "empty-case-tray"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, required=True, help="COCO export zip")
    parser.add_argument("--output", type=Path, default=Path("battery_yolo_dataset"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument(
        "--min-contour-area",
        type=float,
        default=8.0,
        help="Discard tiny polygon fragments after converting masks",
    )
    return parser.parse_args()


def find_coco_json(root: Path) -> Path:
    candidates = list(root.rglob("*_annotations.coco.json")) + list(
        root.rglob("_annotations.coco.json")
    )
    candidates = sorted(set(candidates))
    if not candidates:
        raise FileNotFoundError("No COCO annotation JSON was found in the zip.")
    if len(candidates) > 1:
        print(f"Found {len(candidates)} COCO files; using {candidates[0]}")
    return candidates[0]


def decode_annotation_mask(annotation: dict[str, Any], height: int, width: int) -> np.ndarray:
    """Decode polygon or RLE COCO segmentation into a binary mask."""
    segmentation = annotation.get("segmentation")
    if not segmentation:
        return np.zeros((height, width), dtype=np.uint8)

    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(segmentation, dict):
        rle = segmentation
        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, height, width)
    else:
        raise TypeError(f"Unsupported segmentation type: {type(segmentation)}")

    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    return decoded.astype(np.uint8)


def largest_external_contour(mask: np.ndarray, min_area: float) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not contours:
        return None
    return max(contours, key=cv2.contourArea).reshape(-1, 2)


def yolo_detection_line(class_id: int, bbox: list[float], width: int, height: int) -> str:
    x, y, w, h = bbox
    cx = (x + w / 2.0) / width
    cy = (y + h / 2.0) / height
    nw = w / width
    nh = h / height
    return f"{class_id} {cx:.8f} {cy:.8f} {nw:.8f} {nh:.8f}"


def yolo_segment_line(class_id: int, contour: np.ndarray, width: int, height: int) -> str:
    contour = contour.astype(np.float64)
    contour[:, 0] = np.clip(contour[:, 0] / width, 0.0, 1.0)
    contour[:, 1] = np.clip(contour[:, 1] / height, 0.0, 1.0)
    coords = " ".join(f"{value:.8f}" for value in contour.reshape(-1))
    return f"{class_id} {coords}"


def split_images(image_ids: list[int], train_ratio: float, val_ratio: float, seed: int) -> dict[str, set[int]]:
    ids = list(image_ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_train = max(1, round(n * train_ratio))
    n_val = max(1, round(n * val_ratio))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    return {
        "train": set(ids[:n_train]),
        "val": set(ids[n_train : n_train + n_val]),
        "test": set(ids[n_train + n_val :]),
    }


def main() -> None:
    args = parse_args()
    total = args.train_ratio + args.val_ratio + args.test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split ratios must total 1.0, got {total}")

    output = args.output.resolve()
    extract_dir = output / "_coco_export"
    if output.exists():
        shutil.rmtree(output)
    extract_dir.mkdir(parents=True)

    with zipfile.ZipFile(args.zip, "r") as zf:
        zf.extractall(extract_dir)

    json_path = find_coco_json(extract_dir)
    source_image_dir = json_path.parent
    coco = json.loads(json_path.read_text(encoding="utf-8"))

    # Category 0 in this export is a dataset/supercategory placeholder, not an object class.
    categories = [c for c in coco["categories"] if c["id"] != 0]
    categories.sort(key=lambda c: c["id"])
    old_to_new = {category["id"]: i for i, category in enumerate(categories)}
    names = [CLASS_RENAMES.get(category["name"], category["name"]) for category in categories]

    images_by_id = {image["id"]: image for image in coco["images"]}
    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        if annotation["category_id"] in old_to_new:
            anns_by_image[annotation["image_id"]].append(annotation)

    splits = split_images(
        list(images_by_id), args.train_ratio, args.val_ratio, args.seed
    )

    for task in ("detect", "segment"):
        for split in splits:
            (output / task / "images" / split).mkdir(parents=True, exist_ok=True)
            (output / task / "labels" / split).mkdir(parents=True, exist_ok=True)

    class_counts: Counter[str] = Counter()
    skipped_masks = 0

    for split, image_ids in splits.items():
        for image_id in sorted(image_ids):
            info = images_by_id[image_id]
            source = source_image_dir / info["file_name"]
            if not source.exists():
                matches = list(extract_dir.rglob(info["file_name"]))
                if not matches:
                    raise FileNotFoundError(f"Cannot find image {info['file_name']}")
                source = matches[0]

            suffix = source.suffix.lower()
            if suffix not in IMAGE_EXTENSIONS:
                raise ValueError(f"Unsupported image extension: {source}")

            width, height = int(info["width"]), int(info["height"])
            stem = source.stem
            detect_lines: list[str] = []
            segment_lines: list[str] = []

            for annotation in anns_by_image.get(image_id, []):
                new_class = old_to_new[annotation["category_id"]]
                class_counts[names[new_class]] += 1
                detect_lines.append(
                    yolo_detection_line(new_class, annotation["bbox"], width, height)
                )

                mask = decode_annotation_mask(annotation, height, width)
                contour = largest_external_contour(mask, args.min_contour_area)
                if contour is None or len(contour) < 3:
                    skipped_masks += 1
                    continue
                segment_lines.append(
                    yolo_segment_line(new_class, contour, width, height)
                )

            for task, lines in (("detect", detect_lines), ("segment", segment_lines)):
                image_dest = output / task / "images" / split / source.name
                label_dest = output / task / "labels" / split / f"{stem}.txt"
                shutil.copy2(source, image_dest)
                label_dest.write_text("\n".join(lines), encoding="utf-8")

    for task in ("detect", "segment"):
        task_root = output / task
        yaml_data = {
            "path": str(task_root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {index: name for index, name in enumerate(names)},
        }
        (task_root / "data.yaml").write_text(
            yaml.safe_dump(yaml_data, sort_keys=False), encoding="utf-8"
        )

    summary = {
        "images": len(images_by_id),
        "annotations": sum(class_counts.values()),
        "splits": {name: len(ids) for name, ids in splits.items()},
        "classes": names,
        "class_counts": dict(class_counts),
        "skipped_segmentation_masks": skipped_masks,
    }
    (output / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    print(f"\nPrepared datasets at: {output}")
    print(f"Detection YAML:    {output / 'detect' / 'data.yaml'}")
    print(f"Segmentation YAML: {output / 'segment' / 'data.yaml'}")


if __name__ == "__main__":
    main()
