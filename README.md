# Road Damage Detection and Severity Assessment — RDD2022 India

This implementation is adapted to the uploaded `archive.zip`.

## Dataset found in the uploaded archive

The archive contains:

- 1,530 JPG road images
- 1,530 Pascal-VOC XML annotation files
- 4,524 annotated objects
- RDD2022 damage codes including D00, D10, D20, D40 and additional codes.

The project brief asks for exactly four classes, so the model uses:

| RDD code | Model class |
|---|---|
| D00 | Longitudinal crack |
| D10 | Transverse crack |
| D20 | Alligator crack |
| D40 | Pothole |

Other RDD labels in the archive are intentionally ignored.

## Important accuracy limitation

The supplied data contains damage-location/type labels, but it does not contain Low/Medium/High severity ground-truth labels.

Therefore:

- defect detection/classification is learned from RDD2022;
- severity is estimated with a transparent heuristic;
- do NOT report a severity accuracy until severity labels are collected.

The rarest requested class is D10, with only 23 boxes in this archive. This imbalance should be reported when presenting results.

## Setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

## Prepare the supplied training folder

```bash
python scripts/prepare_archive.py ^
    --source-dir train ^
    --output prepared ^
    --val-ratio 0.20
```

Linux/macOS:

```bash
python scripts/prepare_archive.py --source-dir train --output prepared --val-ratio 0.20
```

The converter reads `train/images` and `train/annotations/xmls`, creates a reproducible train/validation split, and writes YOLO-format data plus `prepared/data.yaml`. ZIP archives are also supported with `--archive archive.zip`.

## Train the detector

A lightweight YOLO model is recommended:

```bash
python scripts/train.py     --data prepared/data.yaml     --model yolo11n.pt     --epochs 100     --imgsz 640     --batch 16
```

For a quick CPU run, use a smaller dataset fraction and image size:

```bash
python scripts/train.py --data prepared/data.yaml --name road_damage_full --epochs 5 --imgsz 320 --batch 16 --device cpu
```

For a GPU with more memory, try:

```bash
python scripts/train.py     --data prepared/data.yaml     --model yolo11s.pt     --epochs 150     --imgsz 960     --batch 8
```

## Evaluate

```bash
python scripts/evaluate.py     --weights runs/road_damage/weights/best.pt     --data prepared/data.yaml
```

The detector should be reported using:

- Precision
- Recall
- mAP@0.50
- mAP@0.50:0.95
- per-class AP

Do not use training accuracy as the main model-quality metric for object detection.

## Run inference

```bash
python scripts/infer.py     --weights runs/road_damage/weights/best.pt     --source test_image.jpg     --output predictions
```

The CSV contains:

- image
- defect class
- confidence
- severity
- severity score
- bounding-box coordinates

## Optional web UI

After training:

```bash
streamlit run app/app.py
```

Upload a road image and the UI displays:

- annotated image
- defect type
- confidence
- severity
- bounding box coordinates

## How to make severity genuinely accurate

Create a severity-labelled subset with labels:

```text
image_id, object_id, severity
India_000005, 1, Low
India_000017, 1, High
...
```

Then train a separate severity classifier using the detected crop. The recommended architecture is:

1. YOLO detector → bounding box + damage type
2. Crop detected damage
3. Severity CNN/classifier → Low / Medium / High
4. Evaluate severity using macro-F1, balanced accuracy and confusion matrix.

This is preferable to pretending that severity can be learned from RDD2022 type/bounding-box annotations alone.
