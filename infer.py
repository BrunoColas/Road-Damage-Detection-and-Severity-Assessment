import argparse
import csv
from pathlib import Path
import cv2
from ultralytics import YOLO
from severity import estimate_severity

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", default="predictions")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.50)
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=640,
        verbose=False
    )

    rows = []

    for result in results:
        img = result.orig_img.copy()
        h, w = img.shape[:2]

        if result.boxes is not None:
            for box in result.boxes:
                cls = int(box.cls.item())
                confidence = float(box.conf.item())
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                name = model.names[cls]

                severity, score = estimate_severity(
                    name, x1, y1, x2, y2, w, h
                )

                cv2.rectangle(
                    img,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2
                )

                label = f"{name} | {confidence:.2f} | {severity}"
                cv2.putText(
                    img, label,
                    (int(x1), max(25, int(y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2
                )

                rows.append({
                    "image": Path(result.path).name,
                    "class": name,
                    "confidence": round(confidence, 4),
                    "severity": severity,
                    "severity_score": score,
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2)
                })

        cv2.imwrite(str(out / Path(result.path).name), img)

    csv_path = out / "detections.csv"
    fields = [
        "image", "class", "confidence", "severity",
        "severity_score", "x1", "y1", "x2", "y2"
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", out.resolve())
    print("CSV:", csv_path.resolve())

if __name__ == "__main__":
    main()
