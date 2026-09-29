import argparse
import random
import shutil
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
import yaml

TARGET = {"D00": 0, "D10": 1, "D20": 2, "D40": 3}
NAMES = ["Longitudinal crack", "Transverse crack", "Alligator crack", "Pothole"]

def parse_xml(xml_bytes):
    root = ET.fromstring(xml_bytes)
    size = root.find("size")
    w = float(size.findtext("width"))
    h = float(size.findtext("height"))
    labels = []
    for obj in root.findall("object"):
        code = (obj.findtext("name") or "").strip().upper()
        if code not in TARGET:
            continue
        bb = obj.find("bndbox")
        x1 = max(0, min(float(bb.findtext("xmin")), w - 1))
        y1 = max(0, min(float(bb.findtext("ymin")), h - 1))
        x2 = max(0, min(float(bb.findtext("xmax")), w))
        y2 = max(0, min(float(bb.findtext("ymax")), h))
        if x2 <= x1 or y2 <= y1:
            continue
        labels.append((TARGET[code], x1, y1, x2, y2, w, h))
    return labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", help="ZIP archive containing images and VOC XML files")
    ap.add_argument("--source-dir", help="Unpacked dataset root, such as train/")
    ap.add_argument("--output", required=True)
    ap.add_argument("--val-ratio", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if bool(args.archive) == bool(args.source_dir):
        ap.error("provide exactly one of --archive or --source-dir")

    out = Path(args.output)
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)

    archive = zipfile.ZipFile(args.archive) if args.archive else None
    if archive:
        names = archive.namelist()
        xmls = [n for n in names if n.lower().endswith(".xml")]
        image_lookup = {
            Path(n).name: n for n in names
            if Path(n).suffix.lower() in {".jpg", ".jpeg", ".png"}
        }
        read_bytes = archive.read
    else:
        source = Path(args.source_dir)
        images_dir = source / "images"
        annotations_dir = source / "annotations" / "xmls"
        if not images_dir.is_dir() or not annotations_dir.is_dir():
            ap.error("--source-dir must contain images/ and annotations/xmls/")
        xmls = sorted(annotations_dir.glob("*.xml"))
        image_lookup = {
            path.name: path for path in images_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        }
        read_bytes = lambda path: Path(path).read_bytes()

    records = []
    total_codes = Counter()

    for xn in xmls:
        xml_bytes = read_bytes(xn)
        root = ET.fromstring(xml_bytes)
        filename = root.findtext("filename") or (Path(xn).stem + ".jpg")
        image_path = image_lookup.get(Path(filename).name)
        if image_path is None:
            continue

        labels = parse_xml(xml_bytes)
        for label in labels:
            total_codes[label[0]] += 1
        if labels:
            records.append((xn, image_path, labels))

    rng = random.Random(args.seed)
    rng.shuffle(records)
    val_count = max(1, round(len(records) * args.val_ratio))

    # Force a reproducible representation of the rare D10 class in validation.
    val = []
    d10 = [r for r in records if any(x[0] == 1 for x in r[2])]
    rng.shuffle(d10)
    for record in d10[:max(1, round(len(d10) * args.val_ratio))]:
        val.append(record)

    val_ids = {r[0] for r in val}
    remaining = [r for r in records if r[0] not in val_ids]
    val.extend(remaining[:max(0, val_count - len(val))])
    val_ids = {r[0] for r in val}

    splits = {
        "val": val,
        "train": [r for r in records if r[0] not in val_ids]
    }

    for split, subset in splits.items():
        for _, image_path, labels in subset:
            stem = Path(image_path).stem
            img_dst = out / "images" / split / Path(image_path).name
            lab_dst = out / "labels" / split / f"{stem}.txt"

            img_dst.write_bytes(read_bytes(image_path))

            lines = []
            for cls, x1, y1, x2, y2, w, h in labels:
                xc = ((x1 + x2) / 2) / w
                yc = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                lines.append(
                    f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
                )
            lab_dst.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if archive:
        archive.close()

    data = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: n for i, n in enumerate(NAMES)}
    }
    (out / "data.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8"
    )

    print("Prepared dataset")
    print("Images:", len(records))
    print("Train:", len(splits["train"]))
    print("Validation:", len(splits["val"]))
    print("Target object counts:", {
        NAMES[k]: v for k, v in sorted(total_codes.items())
    })

if __name__ == "__main__":
    main()
