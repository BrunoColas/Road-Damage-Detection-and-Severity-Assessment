import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--name", default="road_damage")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--fraction", type=float, default=1.0)
    args = ap.parse_args()

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(Path(__file__).resolve().parents[1] / "runs"),
        name=args.name,
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
        fraction=args.fraction,
    )

if __name__ == "__main__":
    main()
