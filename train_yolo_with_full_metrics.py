"""Train and evaluate a local Ultralytics YOLO model with live and saved monitoring.

This version adds three levels of monitoring:

1. Ultralytics' normal output files, including ``results.csv``, ``results.png``,
   confusion matrices and confidence curves.
2. TensorBoard for live monitoring while training is running.
3. A self-contained Plotly HTML report after test-set validation, containing
   losses, precision/recall/mAP, instance-level TP/FP/FN counts, image-level
   TP/FP/FN/TN presence metrics, confusion matrices and confidence-score
   distributions.

Examples
--------
Recommended segmentation baseline::

    python train_yolo.py --task segment \
        --data battery_yolo_dataset/segment/data.yaml \
        --epochs 150 --imgsz 640 --batch 4 --device 0

Detection-only model for a later YOLO + SAM pipeline::

    python train_yolo.py --task detect \
        --data battery_yolo_dataset/detect/data.yaml \
        --epochs 150 --imgsz 640 --batch 4 --device 0

While training is running, open another terminal and run::

    tensorboard --logdir runs/battery

Then open http://localhost:6006 in a browser.
"""


#this can be run for evaluation and results only with this terminal command :
#

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLO and save live plus post-training diagnostics."
    )
    parser.add_argument("--task", choices=("detect", "segment"), default="segment")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default=None, help="Pretrained checkpoint or trained best.pt")
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help=(
            "Skip training and evaluate an existing --model checkpoint. Useful when "
            "training is already complete and you only need TP/FP/FN/TN reports."
        ),
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default=None, help="Examples: 0, cpu, mps")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--project", default="runs/battery")
    parser.add_argument("--name", default=None)
    parser.add_argument(
        "--save-period",
        type=int,
        default=10,
        help="Save an additional checkpoint every N epochs. Use -1 to disable.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="Dataset split used for the final detailed report.",
    )
    parser.add_argument(
        "--val-conf",
        type=float,
        default=0.001,
        help=(
            "Low validation confidence preserves the full precision-recall curve. "
            "The confusion matrix still uses Ultralytics' matching logic."
        ),
    )
    parser.add_argument(
        "--presence-conf",
        type=float,
        default=0.25,
        help=(
            "Confidence threshold used for image-level class-presence TP/FP/FN/TN. "
            "This is separate from --val-conf, which is intentionally low for PR curves."
        ),
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.7,
        help="IoU used by non-maximum suppression during final validation.",
    )
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument(
        "--report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create CSV/JSON files and an interactive Plotly HTML report.",
    )
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    """Convert NumPy and Path values into JSON-safe Python values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _concat_metric_stats(metrics: Any) -> dict[str, np.ndarray]:
    """Concatenate Ultralytics validation statistics without assuming a version."""
    output: dict[str, np.ndarray] = {}
    stats = getattr(metrics, "stats", {}) or {}

    for key, values in stats.items():
        arrays: list[np.ndarray] = []
        for value in values:
            array = np.asarray(value)
            if array.size:
                arrays.append(array)
        if not arrays:
            continue
        try:
            output[key] = np.concatenate(arrays, axis=0)
        except ValueError:
            output[key] = np.asarray(arrays, dtype=object)
    return output


def _class_names(metrics: Any) -> list[str]:
    names = getattr(metrics, "names", {})
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]
    return [str(name) for name in names]


def _counts_from_confusion_matrix(metrics: Any) -> pd.DataFrame:
    """Return TP, FP and FN counts per class from the validation matrix.

    Ultralytics' detection matrix uses rows for predicted classes and columns
    for true classes. The final row and column represent background.
    """
    confusion = getattr(metrics, "confusion_matrix", None)
    names = _class_names(metrics)
    if confusion is None or not hasattr(confusion, "matrix"):
        return pd.DataFrame(columns=["Class", "TP", "FP", "FN"])

    matrix = np.asarray(confusion.matrix, dtype=float)
    class_count = len(names)
    if matrix.ndim != 2 or matrix.shape[0] < class_count or matrix.shape[1] < class_count:
        return pd.DataFrame(columns=["Class", "TP", "FP", "FN"])

    true_positive = np.diag(matrix)[:class_count]
    false_positive = matrix[:class_count, :].sum(axis=1) - true_positive
    false_negative = matrix[:, :class_count].sum(axis=0) - true_positive

    return pd.DataFrame(
        {
            "Class": names,
            "TP": true_positive.astype(int),
            "FP": false_positive.astype(int),
            "FN": false_negative.astype(int),
        }
    )


def _confidence_dataframe(metrics: Any) -> pd.DataFrame:
    """Return prediction confidence and TP/FP status at IoU 0.50.

    For segmentation models the table also includes mask TP/FP status when
    Ultralytics exposes ``tp_m`` in the validation statistics.
    """
    stats = _concat_metric_stats(metrics)
    required = {"conf", "pred_cls", "tp"}
    if not required.issubset(stats):
        return pd.DataFrame(
            columns=["Confidence", "Class", "Box outcome", "Mask outcome"]
        )

    confidences = np.asarray(stats["conf"]).reshape(-1)
    predicted_classes = np.asarray(stats["pred_cls"]).astype(int).reshape(-1)
    box_tp_array = np.asarray(stats["tp"])
    if box_tp_array.ndim == 1:
        box_true = box_tp_array.astype(bool)
    else:
        box_true = box_tp_array[:, 0].astype(bool)  # first column is IoU 0.50

    names = _class_names(metrics)
    class_labels = [
        names[index] if 0 <= index < len(names) else f"class_{index}"
        for index in predicted_classes
    ]

    data: dict[str, Any] = {
        "Confidence": confidences,
        "Class": class_labels,
        "Box outcome": np.where(box_true, "TP", "FP"),
    }

    if "tp_m" in stats:
        mask_tp_array = np.asarray(stats["tp_m"])
        if mask_tp_array.ndim == 1:
            mask_true = mask_tp_array.astype(bool)
        else:
            mask_true = mask_tp_array[:, 0].astype(bool)
        data["Mask outcome"] = np.where(mask_true, "TP", "FP")
    else:
        data["Mask outcome"] = "Not applicable"

    length = min(len(np.asarray(value)) for value in data.values())
    return pd.DataFrame({key: np.asarray(value)[:length] for key, value in data.items()})


def _training_results_dataframe(train_dir: Path) -> pd.DataFrame:
    csv_path = train_dir / "results.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    dataframe = pd.read_csv(csv_path)
    dataframe.columns = [column.strip() for column in dataframe.columns]
    return dataframe


def _summary_dataframe(metrics: Any) -> pd.DataFrame:
    try:
        summary = metrics.summary(decimals=5)
    except Exception:
        summary = []
    return pd.DataFrame(summary)


def _safe_divide(numerator: float, denominator: float) -> float:
    """Return a finite ratio, using 0.0 when the denominator is zero."""
    return float(numerator / denominator) if denominator else 0.0


def _add_instance_metrics(counts: pd.DataFrame) -> pd.DataFrame:
    """Add standard detection metrics to instance-level TP/FP/FN counts.

    True negatives are intentionally not added here. In object detection there is
    no finite, canonical number of background objects that were correctly not
    detected, so instance-level TN is undefined. TN is computed separately below
    at image/class-presence level, where every image-class pair is a binary case.
    """
    if counts.empty:
        return counts

    output = counts.copy()
    output["Precision"] = [
        _safe_divide(tp, tp + fp)
        for tp, fp in zip(output["TP"], output["FP"])
    ]
    output["Recall"] = [
        _safe_divide(tp, tp + fn)
        for tp, fn in zip(output["TP"], output["FN"])
    ]
    output["F1"] = [
        _safe_divide(2 * tp, 2 * tp + fp + fn)
        for tp, fp, fn in zip(output["TP"], output["FP"], output["FN"])
    ]
    return output


def _resolve_dataset_root(data_yaml: Path, config: dict[str, Any]) -> Path:
    """Resolve the Ultralytics dataset root from a data YAML file."""
    raw_root = config.get("path")
    if raw_root is None:
        return data_yaml.parent.resolve()
    root = Path(str(raw_root)).expanduser()
    if not root.is_absolute():
        root = data_yaml.parent / root
    return root.resolve()


def _image_files_from_entry(entry: Any, dataset_root: Path) -> list[Path]:
    """Expand a YAML split entry into image files."""
    if isinstance(entry, (list, tuple)):
        images: list[Path] = []
        for item in entry:
            images.extend(_image_files_from_entry(item, dataset_root))
        return sorted(set(images))

    source = Path(str(entry)).expanduser()
    if not source.is_absolute():
        source = dataset_root / source
    source = source.resolve()

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    if source.is_dir():
        return sorted(
            path for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in image_extensions
        )
    if source.is_file() and source.suffix.lower() == ".txt":
        images: list[Path] = []
        for line in source.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            path = Path(line).expanduser()
            if not path.is_absolute():
                path = dataset_root / path
            images.append(path.resolve())
        return images
    if source.is_file() and source.suffix.lower() in image_extensions:
        return [source]
    raise FileNotFoundError(f"Could not resolve dataset split entry: {source}")


def _label_path_for_image(image_path: Path) -> Path:
    """Map a YOLO image path to the corresponding label text file."""
    parts = list(image_path.parts)
    image_indices = [i for i, part in enumerate(parts) if part.lower() == "images"]
    if image_indices:
        parts[image_indices[-1]] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def _ground_truth_classes(label_path: Path) -> set[int]:
    """Read class IDs present in a YOLO detection or segmentation label file."""
    if not label_path.exists():
        return set()
    class_ids: set[int] = set()
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        try:
            class_ids.add(int(float(fields[0])))
        except ValueError:
            continue
    return class_ids


def _image_level_presence_metrics(
    model: YOLO,
    data_yaml: Path,
    split: str,
    names: list[str],
    confidence_threshold: float,
    imgsz: int,
    device: str | None,
    nms_iou: float,
    max_det: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute TP/FP/FN/TN for class presence in each image.

    Each image-class pair is treated as a binary classification case:

    * TP: the class exists in the labels and is predicted at least once.
    * FP: the class does not exist but is predicted at least once.
    * FN: the class exists but is not predicted.
    * TN: the class does not exist and is not predicted.

    These are valid TNs, but they answer an image-level presence question rather
    than an instance-localisation question.
    """
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    if split not in config:
        raise KeyError(f"Split '{split}' is not defined in {data_yaml}")
    dataset_root = _resolve_dataset_root(data_yaml, config)
    image_paths = _image_files_from_entry(config[split], dataset_root)
    if not image_paths:
        raise ValueError(f"No images found for split '{split}' in {data_yaml}")

    prediction_by_path: dict[Path, set[int]] = {}
    results = model.predict(
        source=[str(path) for path in image_paths],
        conf=confidence_threshold,
        iou=nms_iou,
        imgsz=imgsz,
        device=device,
        max_det=max_det,
        verbose=False,
        stream=True,
    )
    for result in results:
        result_path = Path(str(result.path)).resolve()
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.cls is None:
            prediction_by_path[result_path] = set()
        else:
            prediction_by_path[result_path] = {
                int(value) for value in boxes.cls.detach().cpu().numpy().tolist()
            }

    records: list[dict[str, Any]] = []
    per_image_records: list[dict[str, Any]] = []
    class_count = len(names)
    counters = {
        class_id: {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
        for class_id in range(class_count)
    }

    for image_path in image_paths:
        resolved = image_path.resolve()
        truth = _ground_truth_classes(_label_path_for_image(resolved))
        predicted = prediction_by_path.get(resolved, set())
        for class_id, class_name in enumerate(names):
            actual_present = class_id in truth
            predicted_present = class_id in predicted
            if actual_present and predicted_present:
                outcome = "TP"
            elif not actual_present and predicted_present:
                outcome = "FP"
            elif actual_present and not predicted_present:
                outcome = "FN"
            else:
                outcome = "TN"
            counters[class_id][outcome] += 1
            per_image_records.append(
                {
                    "Image": str(resolved),
                    "Class": class_name,
                    "Actual present": actual_present,
                    "Predicted present": predicted_present,
                    "Outcome": outcome,
                }
            )

    for class_id, class_name in enumerate(names):
        tp = counters[class_id]["TP"]
        fp = counters[class_id]["FP"]
        fn = counters[class_id]["FN"]
        tn = counters[class_id]["TN"]
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        specificity = _safe_divide(tn, tn + fp)
        npv = _safe_divide(tn, tn + fn)
        accuracy = _safe_divide(tp + tn, tp + fp + fn + tn)
        f1 = _safe_divide(2 * tp, 2 * tp + fp + fn)
        balanced_accuracy = (recall + specificity) / 2.0
        denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = _safe_divide(tp * tn - fp * fn, denominator)
        records.append(
            {
                "Class": class_name,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "Precision": precision,
                "Recall/Sensitivity": recall,
                "Specificity": specificity,
                "NPV": npv,
                "Accuracy": accuracy,
                "F1": f1,
                "Balanced accuracy": balanced_accuracy,
                "MCC": mcc,
                "Images": len(image_paths),
            }
        )

    per_class = pd.DataFrame(records)
    per_image = pd.DataFrame(per_image_records)

    total_tp = int(per_class["TP"].sum())
    total_fp = int(per_class["FP"].sum())
    total_fn = int(per_class["FN"].sum())
    total_tn = int(per_class["TN"].sum())
    micro_precision = _safe_divide(total_tp, total_tp + total_fp)
    micro_recall = _safe_divide(total_tp, total_tp + total_fn)
    micro_specificity = _safe_divide(total_tn, total_tn + total_fp)
    micro_accuracy = _safe_divide(
        total_tp + total_tn, total_tp + total_fp + total_fn + total_tn
    )
    micro_f1 = _safe_divide(2 * total_tp, 2 * total_tp + total_fp + total_fn)
    micro_denominator = math.sqrt(
        (total_tp + total_fp)
        * (total_tp + total_fn)
        * (total_tn + total_fp)
        * (total_tn + total_fn)
    )
    micro_mcc = _safe_divide(
        total_tp * total_tn - total_fp * total_fn, micro_denominator
    )
    overall = pd.DataFrame(
        [
            {
                "Averaging": "Micro",
                "TP": total_tp,
                "FP": total_fp,
                "FN": total_fn,
                "TN": total_tn,
                "Precision": micro_precision,
                "Recall/Sensitivity": micro_recall,
                "Specificity": micro_specificity,
                "Accuracy": micro_accuracy,
                "F1": micro_f1,
                "Balanced accuracy": (micro_recall + micro_specificity) / 2.0,
                "MCC": micro_mcc,
            },
            {
                "Averaging": "Macro",
                "TP": np.nan,
                "FP": np.nan,
                "FN": np.nan,
                "TN": np.nan,
                "Precision": float(per_class["Precision"].mean()),
                "Recall/Sensitivity": float(per_class["Recall/Sensitivity"].mean()),
                "Specificity": float(per_class["Specificity"].mean()),
                "Accuracy": float(per_class["Accuracy"].mean()),
                "F1": float(per_class["F1"].mean()),
                "Balanced accuracy": float(per_class["Balanced accuracy"].mean()),
                "MCC": float(per_class["MCC"].mean()),
            },
        ]
    )
    return per_class, overall, per_image


def _plotly_dashboard(
    output_path: Path,
    training: pd.DataFrame,
    class_summary: pd.DataFrame,
    counts: pd.DataFrame,
    presence: pd.DataFrame,
    presence_overall: pd.DataFrame,
    confidence: pd.DataFrame,
    metrics: Any,
    run_name: str,
) -> None:
    """Build one self-contained interactive HTML report using Plotly."""
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError(
            "Plotly is not installed. Run: pip install plotly"
        ) from exc

    figures: list[tuple[str, Any]] = []

    if not training.empty:
        epoch_column = "epoch" if "epoch" in training.columns else training.columns[0]
        loss_columns = [column for column in training.columns if "loss" in column.lower()]
        metric_columns = [
            column
            for column in training.columns
            if any(token in column.lower() for token in ("precision", "recall", "map"))
        ]
        learning_rate_columns = [
            column for column in training.columns if column.lower().startswith("lr/")
        ]

        if loss_columns:
            figure = px.line(
                training,
                x=epoch_column,
                y=loss_columns,
                markers=False,
                title="Training and validation losses",
            )
            figure.update_layout(yaxis_title="Loss")
            figures.append(("Loss curves", figure))

        if metric_columns:
            figure = px.line(
                training,
                x=epoch_column,
                y=metric_columns,
                markers=False,
                title="Validation precision, recall and mAP by epoch",
            )
            figure.update_layout(yaxis_title="Metric value")
            figures.append(("Performance by epoch", figure))

        if learning_rate_columns:
            figure = px.line(
                training,
                x=epoch_column,
                y=learning_rate_columns,
                title="Learning-rate schedule",
            )
            figure.update_layout(yaxis_title="Learning rate")
            figures.append(("Learning rate", figure))

    if not class_summary.empty and "Class" in class_summary.columns:
        candidate_columns = [
            column
            for column in class_summary.columns
            if column not in {"Class", "Images", "Instances"}
            and pd.api.types.is_numeric_dtype(class_summary[column])
        ]
        if candidate_columns:
            melted = class_summary.melt(
                id_vars="Class",
                value_vars=candidate_columns,
                var_name="Metric",
                value_name="Value",
            )
            figure = px.bar(
                melted,
                x="Class",
                y="Value",
                color="Metric",
                barmode="group",
                title="Per-class validation metrics",
            )
            figure.update_layout(xaxis_tickangle=-35)
            figures.append(("Per-class metrics", figure))

    if not counts.empty:
        melted = counts.melt(
            id_vars="Class",
            value_vars=["TP", "FP", "FN"],
            var_name="Outcome",
            value_name="Count",
        )
        figure = px.bar(
            melted,
            x="Class",
            y="Count",
            color="Outcome",
            barmode="group",
            title="Instance-level detection outcomes",
        )
        figure.update_layout(xaxis_tickangle=-35)
        figures.append(("Instance-level TP, FP and FN", figure))

    if not presence.empty:
        presence_counts = presence.melt(
            id_vars="Class",
            value_vars=["TP", "FP", "FN", "TN"],
            var_name="Outcome",
            value_name="Count",
        )
        figure = px.bar(
            presence_counts,
            x="Class",
            y="Count",
            color="Outcome",
            barmode="group",
            title="Image-level class-presence confusion counts",
        )
        figure.update_layout(xaxis_tickangle=-35)
        figures.append(("Image-level TP, FP, FN and TN", figure))

        metric_columns = [
            "Precision",
            "Recall/Sensitivity",
            "Specificity",
            "Accuracy",
            "F1",
            "Balanced accuracy",
            "MCC",
        ]
        available = [column for column in metric_columns if column in presence.columns]
        if available:
            melted_metrics = presence.melt(
                id_vars="Class",
                value_vars=available,
                var_name="Metric",
                value_name="Value",
            )
            figure = px.bar(
                melted_metrics,
                x="Class",
                y="Value",
                color="Metric",
                barmode="group",
                title="Image-level presence metrics by class",
            )
            figure.update_layout(xaxis_tickangle=-35, yaxis_range=[-1, 1])
            figures.append(("Presence precision, recall, specificity, F1 and related metrics", figure))

    if not confidence.empty:
        box_figure = px.histogram(
            confidence,
            x="Confidence",
            color="Box outcome",
            nbins=25,
            barmode="overlay",
            opacity=0.70,
            marginal="box",
            title="Prediction confidence: box true positives versus false positives",
        )
        figures.append(("Confidence distribution", box_figure))

        if "Mask outcome" in confidence.columns and not (
            confidence["Mask outcome"] == "Not applicable"
        ).all():
            mask_figure = px.histogram(
                confidence,
                x="Confidence",
                color="Mask outcome",
                nbins=25,
                barmode="overlay",
                opacity=0.70,
                marginal="box",
                title="Prediction confidence: mask true positives versus false positives",
            )
            figures.append(("Mask confidence distribution", mask_figure))

    confusion = getattr(metrics, "confusion_matrix", None)
    names = _class_names(metrics)
    if confusion is not None and hasattr(confusion, "matrix"):
        matrix = np.asarray(confusion.matrix, dtype=float)
        labels = [*names, "background"]
        if matrix.shape == (len(labels), len(labels)):
            figure = go.Figure(
                data=go.Heatmap(
                    z=matrix,
                    x=[f"True: {label}" for label in labels],
                    y=[f"Predicted: {label}" for label in labels],
                    hovertemplate="%{y}<br>%{x}<br>Count=%{z}<extra></extra>",
                )
            )
            figure.update_layout(title="Raw confusion matrix")
            figures.append(("Confusion matrix", figure))

    scalar_metrics = _json_safe(getattr(metrics, "results_dict", {}))
    scalar_rows = "".join(
        f"<tr><td>{key}</td><td>{float(value):.5f}</td></tr>"
        for key, value in scalar_metrics.items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    )

    html_sections = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{run_name} training report</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;max-width:1500px;}"
        "table{border-collapse:collapse;margin-bottom:24px;}"
        "td,th{border:1px solid #ccc;padding:6px 10px;text-align:left;}"
        ".plot{margin:24px 0 44px 0;}code{background:#f1f1f1;padding:2px 5px;}</style>",
        "</head><body>",
        f"<h1>{run_name}: YOLO training and evaluation report</h1>",
        "<p>Instance-level object detection uses TP, FP and FN because the number of "
        "correctly ignored background objects is not finite. TN is therefore reported "
        "separately at image/class-presence level, where every image-class pair is a "
        "well-defined binary case.</p>",
        "<h2>Overall metrics</h2>",
        f"<table><tr><th>Metric</th><th>Value</th></tr>{scalar_rows}</table>",
    ]

    first_figure = True
    for heading, figure in figures:
        html_sections.append(f"<div class='plot'><h2>{heading}</h2>")
        html_sections.append(
            figure.to_html(
                full_html=False,
                include_plotlyjs=True if first_figure else False,
                config={"responsive": True, "displaylogo": False},
            )
        )
        html_sections.append("</div>")
        first_figure = False

    if not class_summary.empty:
        html_sections.append("<h2>Ultralytics per-class detection/segmentation values</h2>")
        html_sections.append(class_summary.to_html(index=False, border=0))

    if not counts.empty:
        html_sections.append("<h2>Instance-level detection metrics</h2>")
        html_sections.append(counts.to_html(index=False, border=0, float_format=lambda x: f"{x:.5f}"))

    if not presence_overall.empty:
        html_sections.append("<h2>Image-level presence metrics: overall</h2>")
        html_sections.append(presence_overall.to_html(index=False, border=0, float_format=lambda x: f"{x:.5f}"))

    if not presence.empty:
        html_sections.append("<h2>Image-level presence metrics: per class</h2>")
        html_sections.append(presence.to_html(index=False, border=0, float_format=lambda x: f"{x:.5f}"))

    html_sections.append("</body></html>")
    output_path.write_text("\n".join(html_sections), encoding="utf-8")


def save_detailed_report(
    train_dir: Path,
    validation_dir: Path,
    metrics: Any,
    run_name: str,
    presence: pd.DataFrame,
    presence_overall: pd.DataFrame,
    presence_per_image: pd.DataFrame,
) -> Path:
    """Save machine-readable metrics plus an interactive HTML dashboard."""
    report_dir = validation_dir / "detailed_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    training = _training_results_dataframe(train_dir)
    summary = _summary_dataframe(metrics)
    counts = _add_instance_metrics(_counts_from_confusion_matrix(metrics))
    confidence = _confidence_dataframe(metrics)

    if not training.empty:
        training.to_csv(report_dir / "training_history.csv", index=False)
    if not summary.empty:
        summary.to_csv(report_dir / "per_class_metrics.csv", index=False)
    if not counts.empty:
        counts.to_csv(report_dir / "instance_detection_metrics_by_class.csv", index=False)
    if not presence.empty:
        presence.to_csv(report_dir / "image_level_presence_metrics_by_class.csv", index=False)
    if not presence_overall.empty:
        presence_overall.to_csv(report_dir / "image_level_presence_metrics_overall.csv", index=False)
    if not presence_per_image.empty:
        presence_per_image.to_csv(report_dir / "image_level_presence_outcomes.csv", index=False)
    if not confidence.empty:
        confidence.to_csv(report_dir / "prediction_confidences.csv", index=False)

    confusion = getattr(metrics, "confusion_matrix", None)
    if confusion is not None and hasattr(confusion, "summary"):
        confusion_records = confusion.summary(normalize=False, decimals=5)
        pd.DataFrame(confusion_records).to_csv(
            report_dir / "confusion_matrix_raw.csv", index=False
        )

    metric_payload = {
        "overall": _json_safe(getattr(metrics, "results_dict", {})),
        "speed_ms": _json_safe(getattr(metrics, "speed", {})),
        "curves": _json_safe(getattr(metrics, "curves", [])),
    }
    (report_dir / "overall_metrics.json").write_text(
        json.dumps(metric_payload, indent=2), encoding="utf-8"
    )

    dashboard_path = report_dir / "training_dashboard.html"
    _plotly_dashboard(
        dashboard_path,
        training=training,
        class_summary=summary,
        counts=counts,
        presence=presence,
        presence_overall=presence_overall,
        confidence=confidence,
        metrics=metrics,
        run_name=run_name,
    )
    return dashboard_path


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")

    default_model = "yolo11n-seg.pt" if args.task == "segment" else "yolo11n.pt"
    model_path = args.model or default_model
    run_name = args.name or f"{args.task}_baseline"

    project_dir = Path(args.project)
    print("\nLIVE MONITORING")
    print("Open a second terminal after training starts and run:")
    print(f"  tensorboard --logdir \"{project_dir}\"")
    print("Then open: http://localhost:6006\n")

    if args.evaluate_only:
        if args.model is None:
            raise ValueError("--evaluate-only requires --model path/to/best.pt")
        best_weights = Path(args.model).expanduser().resolve()
        if not best_weights.exists():
            raise FileNotFoundError(f"Checkpoint not found: {best_weights}")
        # Typical Ultralytics layout is <run>/weights/best.pt.
        train_dir = (
            best_weights.parent.parent
            if best_weights.parent.name == "weights"
            else best_weights.parent
        )
    else:
        model = YOLO(model_path)
        train_kwargs: dict[str, Any] = {
            "data": str(args.data),
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "project": str(project_dir),
            "name": run_name,
            "patience": 35,
            "pretrained": True,
            "plots": True,
            "save": True,
            "save_period": args.save_period,
            "seed": 42,
            "verbose": True,
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

        trainer = getattr(model, "trainer", None)
        train_dir = Path(getattr(trainer, "save_dir", project_dir / run_name))
        best_weights = train_dir / "weights" / "best.pt"
        if not best_weights.exists():
            last_weights = train_dir / "weights" / "last.pt"
            if not last_weights.exists():
                raise FileNotFoundError(
                    f"Could not locate best.pt or last.pt under {train_dir / 'weights'}"
                )
            best_weights = last_weights

    print(f"\nRunning detailed validation using: {best_weights}")
    best_model = YOLO(str(best_weights))
    validation_name = f"{run_name}_{args.split}_evaluation"
    metrics = best_model.val(
        data=str(args.data),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        conf=args.val_conf,
        iou=args.nms_iou,
        max_det=args.max_det,
        plots=True,
        save_txt=True,
        save_conf=True,
        verbose=True,
        project=str(project_dir),
        name=validation_name,
    )

    validation_dir = Path(
        getattr(metrics, "save_dir", project_dir / validation_name)
    )

    print("\nOverall final metrics:")
    for key, value in getattr(metrics, "results_dict", {}).items():
        try:
            print(f"  {key}: {float(value):.5f}")
        except (TypeError, ValueError):
            print(f"  {key}: {value}")

    counts = _add_instance_metrics(_counts_from_confusion_matrix(metrics))
    if not counts.empty:
        print("\nInstance-level detection metrics by class:")
        print(counts.to_string(index=False, float_format=lambda value: f"{value:.5f}"))

    print(
        "\nComputing image-level TP, FP, FN and TN at "
        f"confidence >= {args.presence_conf:.3f}..."
    )
    presence, presence_overall, presence_per_image = _image_level_presence_metrics(
        model=best_model,
        data_yaml=args.data.resolve(),
        split=args.split,
        names=_class_names(metrics),
        confidence_threshold=args.presence_conf,
        imgsz=args.imgsz,
        device=args.device,
        nms_iou=args.nms_iou,
        max_det=args.max_det,
    )
    print("\nImage-level class-presence metrics by class:")
    print(presence.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print("\nImage-level class-presence overall metrics:")
    print(presence_overall.to_string(index=False, float_format=lambda value: f"{value:.5f}"))

    if args.report:
        dashboard = save_detailed_report(
            train_dir=train_dir,
            validation_dir=validation_dir,
            metrics=metrics,
            run_name=run_name,
            presence=presence,
            presence_overall=presence_overall,
            presence_per_image=presence_per_image,
        )
        print(f"\nInteractive report: {dashboard.resolve()}")
        print("Open that HTML file in your browser.")

    print(f"\nUltralytics training files: {train_dir.resolve()}")
    print(f"Ultralytics validation files: {validation_dir.resolve()}")


if __name__ == "__main__":
    main()



''' python train_yolo_with_full_metrics.py --evaluate-only --task segment --model runs/battery/segment_baseline/weights/best.pt --data battery_yolo_dataset/segment/data.yaml --split test --imgsz 640 --batch 4 --device 0 --presence-conf 0.25'''