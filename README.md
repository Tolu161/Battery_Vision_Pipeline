# Local battery-board vision pipeline

This project replaces the hosted Roboflow API with local Ultralytics models.

## Pipeline Order 

1. Prepare both local dataset formats.
2. Train the one-model YOLO segmentation baseline.
3. Run it on the video and verify labels, masks and the priority decision.
4. Train the YOLO detector 

## Environment

Python 3.11.

```bash
pip install -r requirements.txt
```
// environment - battery-vision1 - conda environment 

## 1. Convert and split the COCO export

```bash
python prepare_dataset.py \
  --zip "Battery cell case object detecti 2.coco-segmentation.zip" \
  --output battery_yolo_dataset
```
python prepare_dataset.py --zip "Battery cell case object detecti 2.coco-segmentation.zip" --output battery_yolo_dataset

This creates:

- `battery_yolo_dataset/detect/data.yaml`
- `battery_yolo_dataset/segment/data.yaml`

## 2A. Recommended baseline: fine-tune YOLO instance segmentation

```bash
python train_yolo.py \
  --task segment \
  --data battery_yolo_dataset/segment/data.yaml \
  --epochs 150 \
  --imgsz 640 \
  --batch 4 \
  --device 0
```

python train_yolo.py --task segment --data battery_yolo_dataset/segment/data.yaml --epochs 150 --imgsz 640 -batch 4 --device 0


Run the video:

```bash
python battery_video_pipeline.py \
  --backend yolo-seg \
  --yolo-model runs/battery/segment_baseline/weights/best.pt \
  --video your_video.mp4 \
  --frame-stride 5
```

## 2B. Two-stage alternative: fine-tuned YOLO detector + pretrained SAM 2

```bash
python train_yolo.py \
  --task detect \
  --data battery_yolo_dataset/detect/data.yaml \
  --epochs 150 \
  --imgsz 640 \
  --batch 4 \
  --device 0
```

Then:

```bash
python battery_video_pipeline.py \
  --backend yolo-sam \
  --yolo-model runs/battery/detect_baseline/weights/best.pt \
  --sam-model sam2_t.pt \
  --video your_video.mp4 \
  --frame-stride 5
```

The YOLO model supplies the class and bounding box. SAM uses the box as a prompt to produce the mask.

## Outputs

- `battery_predictions.mp4`: annotated video
- `battery_graphs.jsonl`: detections, scene-graph nodes/edges and selected priority target for every processed frame

