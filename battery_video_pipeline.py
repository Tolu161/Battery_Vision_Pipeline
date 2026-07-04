"""Local video pipeline
Two backends are supported:

1. yolo-seg:
   A fine-tuned YOLO segmentation model directly returns class labels, boxes and masks.

2. yolo-sam:
   A fine-tuned YOLO detector returns class-labelled boxes. Pretrained SAM 2 receives
   each YOLO box as a prompt and refines it into a mask.

The script also turns every processed frame into a small scene graph and applies a
rule-based priority selector. This graph is the structured input that a future GNN
or neural-logic reasoner can learn from.


    python battery_video_pipeline.py \
        --backend yolo-seg \
        --yolo-model runs/battery/segment_baseline/weights/best.pt \
        --video test_video.mp4

    python battery_video_pipeline.py \
        --backend yolo-sam \
        --yolo-model runs/battery/detect_baseline/weights/best.pt \
        --sam-model sam2_t.pt \
        --video test_video.mp4 \
        --frame-stride 5
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from ultralytics import SAM, YOLO


DEFAULT_PRIORITY = [
    "cell-battery",
    "pcb",
    "adhesive-glue",
    "battery-case-cover",
    "battery-case",
    "empty-cell-tray",
    "empty-case-tray",
    "tray",
    "hole",
]

CLASS_ALIASES = {"empy-case-tray": "empty-case-tray"}


@dataclass
class Detection:
    instance_id: str
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]
    center_px: list[float]
    area_px: float
    angle_pca_deg: float | None
    mask: np.ndarray

    def serialisable(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("mask")
        return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("yolo-seg", "yolo-sam"), default="yolo-seg")
    parser.add_argument("--yolo-model", type=Path, required=True)
    parser.add_argument("--sam-model", default="sam2_t.pt")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, default=Path("battery_predictions.mp4"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("battery_graphs.jsonl"))
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Process every Nth input frame. Use 5 or 6 for a lightweight demo.",
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--priority", nargs="+", default=DEFAULT_PRIORITY)
    return parser.parse_args()


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape != (height, width):
        mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
    return mask > 0.5


def mask_geometry(mask: np.ndarray, bbox: Iterable[float]) -> tuple[list[float], float, float | None]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        return [(x1 + x2) / 2.0, (y1 + y2) / 2.0], 0.0, None

    center = [float(xs.mean()), float(ys.mean())]
    area = float(len(xs))
    angle: float | None = None
    if len(xs) >= 5:
        points = np.column_stack((xs, ys)).astype(np.float32)
        _, eigenvectors = cv2.PCACompute(points, mean=None)
        vx, vy = eigenvectors[0]
        angle = float(np.degrees(np.arctan2(vy, vx)))
    return center, area, angle


def result_track_ids(result: Any, count: int, frame_index: int) -> list[int]:
    if result.boxes is not None and result.boxes.id is not None:
        return result.boxes.id.int().cpu().tolist()
    # Stable tracking is preferred, but this fallback keeps the pipeline usable.
    return [frame_index * 10_000 + i for i in range(count)]


def detections_from_yolo_seg(
    model: YOLO,
    frame: np.ndarray,
    frame_index: int,
    confidence: float,
    iou: float,
    imgsz: int,
) -> list[Detection]:
    result = model.track(
        frame,
        persist=True,
        conf=confidence,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
        tracker="bytetrack.yaml",
    )[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []

    height, width = frame.shape[:2]
    boxes = result.boxes.xyxy.cpu().numpy()
    class_ids = result.boxes.cls.int().cpu().tolist()
    confidences = result.boxes.conf.cpu().numpy().tolist()
    track_ids = result_track_ids(result, len(boxes), frame_index)
    masks = result.masks.data.cpu().numpy() if result.masks is not None else None

    output: list[Detection] = []
    for i, box in enumerate(boxes):
        if masks is None or i >= len(masks):
            mask = np.zeros((height, width), dtype=bool)
            x1, y1, x2, y2 = box.astype(int)
            mask[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)] = True
        else:
            mask = resize_mask(masks[i], width, height)

        class_id = int(class_ids[i])
        class_name = CLASS_ALIASES.get(result.names[class_id], result.names[class_id])
        center, area, angle = mask_geometry(mask, box)
        track_id = int(track_ids[i])
        output.append(
            Detection(
                instance_id=f"{class_name}:{track_id}",
                track_id=track_id,
                class_id=class_id,
                class_name=class_name,
                confidence=float(confidences[i]),
                bbox_xyxy=[float(value) for value in box],
                center_px=center,
                area_px=area,
                angle_pca_deg=angle,
                mask=mask,
            )
        )
    return output


def sam_masks_from_boxes(sam_model: SAM, frame: np.ndarray, boxes: np.ndarray) -> list[np.ndarray]:
    if len(boxes) == 0:
        return []
    try:
        result = sam_model.predict(source=frame, bboxes=boxes.tolist(), verbose=False)[0]
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            if len(masks) == len(boxes):
                return list(masks)
    except Exception as exc:
        print(f"Batched SAM prompting failed, falling back to one box at a time: {exc}")

    masks: list[np.ndarray] = []
    for box in boxes:
        result = sam_model.predict(source=frame, bboxes=box.tolist(), verbose=False)[0]
        if result.masks is None or len(result.masks.data) == 0:
            masks.append(np.zeros(frame.shape[:2], dtype=np.float32))
        else:
            masks.append(result.masks.data[0].cpu().numpy())
    return masks


def detections_from_yolo_sam(
    detector: YOLO,
    sam_model: SAM,
    frame: np.ndarray,
    frame_index: int,
    confidence: float,
    iou: float,
    imgsz: int,
) -> list[Detection]:
    result = detector.track(
        frame,
        persist=True,
        conf=confidence,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
        tracker="bytetrack.yaml",
    )[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []

    height, width = frame.shape[:2]
    boxes = result.boxes.xyxy.cpu().numpy()
    class_ids = result.boxes.cls.int().cpu().tolist()
    confidences = result.boxes.conf.cpu().numpy().tolist()
    track_ids = result_track_ids(result, len(boxes), frame_index)
    raw_masks = sam_masks_from_boxes(sam_model, frame, boxes)

    output: list[Detection] = []
    for i, box in enumerate(boxes):
        mask = resize_mask(raw_masks[i], width, height)
        class_id = int(class_ids[i])
        class_name = CLASS_ALIASES.get(result.names[class_id], result.names[class_id])
        center, area, angle = mask_geometry(mask, box)
        track_id = int(track_ids[i])
        output.append(
            Detection(
                instance_id=f"{class_name}:{track_id}",
                track_id=track_id,
                class_id=class_id,
                class_name=class_name,
                confidence=float(confidences[i]),
                bbox_xyxy=[float(value) for value in box],
                center_px=center,
                area_px=area,
                angle_pca_deg=angle,
                mask=mask,
            )
        )
    return output


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(intersection / union) if union else 0.0


def containment(inner: np.ndarray, outer: np.ndarray) -> float:
    inner_area = inner.sum()
    if inner_area == 0:
        return 0.0
    return float(np.logical_and(inner, outer).sum() / inner_area)


def build_scene_graph(detections: list[Detection], frame_shape: tuple[int, ...]) -> dict[str, Any]:
    height, width = frame_shape[:2]
    diagonal = math.hypot(width, height)
    nodes = [
        {
            "id": detection.instance_id,
            "type": detection.class_name,
            "confidence": detection.confidence,
            "center_px": detection.center_px,
            "area_px": detection.area_px,
            "angle_pca_deg": detection.angle_pca_deg,
            "bbox_xyxy": detection.bbox_xyxy,
        }
        for detection in detections
    ]
    edges: list[dict[str, Any]] = []

    for i, first in enumerate(detections):
        for second in detections[i + 1 :]:
            c12 = containment(first.mask, second.mask)
            c21 = containment(second.mask, first.mask)
            overlap = mask_iou(first.mask, second.mask)
            distance = math.dist(first.center_px, second.center_px) / max(diagonal, 1.0)

            if c12 >= 0.80:
                edges.append({"source": first.instance_id, "relation": "inside", "target": second.instance_id, "score": c12})
            elif c21 >= 0.80:
                edges.append({"source": second.instance_id, "relation": "inside", "target": first.instance_id, "score": c21})
            elif overlap >= 0.05:
                edges.append({"source": first.instance_id, "relation": "overlaps", "target": second.instance_id, "score": overlap})
            elif distance <= 0.10:
                edges.append({"source": first.instance_id, "relation": "near", "target": second.instance_id, "score": 1.0 - distance})

    return {"nodes": nodes, "edges": edges}


def select_priority_target(detections: list[Detection], priority: list[str]) -> Detection | None:
    ranks = {name: index for index, name in enumerate(priority)}
    candidates = [d for d in detections if d.class_name in ranks]
    if not candidates:
        return None
    return min(candidates, key=lambda d: (ranks[d.class_name], -d.confidence, -d.area_px))


def colour_for_class(class_id: int) -> tuple[int, int, int]:
    # Deterministic BGR colour without maintaining a hard-coded palette.
    rng = np.random.default_rng(class_id + 11)
    colour = rng.integers(60, 240, size=3)
    return int(colour[0]), int(colour[1]), int(colour[2])


def draw_detections(frame: np.ndarray, detections: list[Detection], selected: Detection | None) -> np.ndarray:
    output = frame.copy()
    overlay = frame.copy()

    for detection in detections:
        colour = colour_for_class(detection.class_id)
        overlay[detection.mask] = (
            0.55 * overlay[detection.mask] + 0.45 * np.asarray(colour)
        ).astype(np.uint8)

    output = cv2.addWeighted(overlay, 0.75, output, 0.25, 0)
    for detection in detections:
        colour = colour_for_class(detection.class_id)
        x1, y1, x2, y2 = [int(value) for value in detection.bbox_xyxy]
        thickness = 4 if selected and detection.instance_id == selected.instance_id else 2
        cv2.rectangle(output, (x1, y1), (x2, y2), colour, thickness)
        label = f"{detection.class_name} {detection.confidence:.2f} id={detection.track_id}"
        cv2.putText(output, label, (x1, max(20, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2, cv2.LINE_AA)
        cx, cy = [int(value) for value in detection.center_px]
        cv2.circle(output, (cx, cy), 4, colour, -1)

    if selected is None:
        decision = "Decision: no priority target detected"
    else:
        decision = f"Decision: target {selected.class_name} (track {selected.track_id})"
    cv2.rectangle(output, (0, 0), (output.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(output, decision, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return output


def main() -> None:
    args = parse_args()
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be at least 1")

    yolo_model = YOLO(str(args.yolo_model))
    sam_model = SAM(args.sam_model) if args.backend == "yolo-sam" else None

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_fps = source_fps / args.frame_stride

    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {args.output_video}")

    frame_index = 0
    processed_index = 0
    with args.output_jsonl.open("w", encoding="utf-8") as jsonl:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % args.frame_stride != 0:
                frame_index += 1
                continue

            if args.backend == "yolo-seg":
                detections = detections_from_yolo_seg(
                    yolo_model, frame, frame_index, args.confidence, args.iou, args.imgsz
                )
            else:
                assert sam_model is not None
                detections = detections_from_yolo_sam(
                    yolo_model,
                    sam_model,
                    frame,
                    frame_index,
                    args.confidence,
                    args.iou,
                    args.imgsz,
                )

            graph = build_scene_graph(detections, frame.shape)
            selected = select_priority_target(detections, args.priority)
            record = {
                "source_frame_index": frame_index,
                "processed_frame_index": processed_index,
                "timestamp_s": frame_index / source_fps,
                "detections": [d.serialisable() for d in detections],
                "graph": graph,
                "decision": None if selected is None else {
                    "target": selected.instance_id,
                    "class_name": selected.class_name,
                    "reason": f"highest available class in priority list {args.priority}",
                },
            }
            jsonl.write(json.dumps(record) + "\n")

            annotated = draw_detections(frame, detections, selected)
            writer.write(annotated)
            print(
                f"frame={frame_index:05d} detections={len(detections):02d} "
                f"target={selected.instance_id if selected else 'None'} edges={len(graph['edges'])}"
            )

            if args.show:
                cv2.imshow("Battery vision pipeline", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1
            processed_index += 1

    capture.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    print(f"Saved annotated video: {args.output_video}")
    print(f"Saved per-frame graphs: {args.output_jsonl}")


if __name__ == "__main__":
    main()
