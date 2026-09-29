import argparse
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        device=args.device,
        plots=True,
        verbose=True
    )

    print("\nDetection metrics")
    print(f"Precision:    {metrics.box.mp:.4f}")
    print(f"Recall:       {metrics.box.mr:.4f}")
    print(f"mAP50:        {metrics.box.map50:.4f}")
    print(f"mAP50-95:     {metrics.box.map:.4f}")

    print("\nPer-class mAP50-95")
    for name, ap in zip(model.names.values(), metrics.box.maps):
        print(f"{name:20s} {ap:.4f}")

if __name__ == "__main__":
    main()
