import argparse
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO


def detect_pothole_index(names):
    for idx, name in names.items():
        if "pothole" in str(name).lower():
            return int(idx)
    mapping = {"d40": 3, "pothole": 0}
    for idx, name in names.items():
        if str(name).lower() in mapping:
            return int(idx)
    return 0


def filter_pothole_dataset(data_path, output_dir):
    data_path = Path(data_path)
    with data_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    source_root = data_path.parent
    dataset_root = Path(output_dir)
    if dataset_root.exists():
        shutil.rmtree(dataset_root)

    for split in ("train", "val"):
        (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    pothole_idx = detect_pothole_index(config.get("names", {}))

    for split in ("train", "val"):
        images_dir = source_root / "images" / split
        labels_dir = source_root / "labels" / split

        if not images_dir.exists() or not labels_dir.exists():
            continue

        for image_path in sorted(images_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue

            boxes = []
            for line in label_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                if cls == pothole_idx:
                    boxes.append(f"0 {parts[1]} {parts[2]} {parts[3]} {parts[4]}")

            if not boxes:
                continue

            target_img = dataset_root / "images" / split / image_path.name
            target_lab = dataset_root / "labels" / split / f"{image_path.stem}.txt"
            shutil.copy2(image_path, target_img)
            target_lab.write_text("\n".join(boxes) + "\n", encoding="utf-8")

    dataset_cfg = {
        "path": str(dataset_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "pothole"},
    }
    (dataset_root / "data.yaml").write_text(
        yaml.safe_dump(dataset_cfg, sort_keys=False),
        encoding="utf-8"
    )

    return dataset_root / "data.yaml"


def main():
    ap = argparse.ArgumentParser(description="Train a pothole-only YOLO detector.")
    ap.add_argument("--data", required=True, help="Path to the prepared data.yaml file")
    ap.add_argument("--model", default="yolo11n.pt", help="Base YOLO model")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--output", default="prepared/pothole_only", help="Directory for filtered pothole dataset")
    args = ap.parse_args()

    pothole_data = filter_pothole_dataset(args.data, args.output)

    model = YOLO(args.model)
    model.train(
        data=str(pothole_data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs",
        name="pothole_detection",
        pretrained=True,
        patience=25,
        cos_lr=True,
        close_mosaic=10,
        degrees=5,
        translate=0.10,
        scale=0.50,
        fliplr=0.5,
        mosaic=0.7,
        mixup=0.05,
        cache=False,
        amp=True,
        workers=4,
    )

    print(f"Pothole dataset created at: {Path(args.output).resolve()}")
    print("Run the app with the default pothole model path: runs/pothole_detection/weights/best.pt")


if __name__ == "__main__":
    main()
