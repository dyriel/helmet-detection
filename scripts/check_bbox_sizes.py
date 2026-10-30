"""
check_bbox_sizes.py

Analyzes bounding box sizes in a YOLO-format dataset to check whether
objects (e.g. helmets) are large enough to be reliably detected at your
training resolution.

Usage:
    python check_bbox_sizes.py --data /path/to/dataset --imgsz 640 --min-px 25

Expects a YOLO folder structure:
    dataset/
        train/images/*.jpg
        train/labels/*.txt   (same filename as image, .txt extension)
        data.yaml            (optional, used to get class names)

Each label line is: class_id x_center y_center width height (all normalized 0-1)
"""

import argparse
import csv
from pathlib import Path
from collections import defaultdict

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Missing dependency. Run: pip install pillow --break-system-packages")

try:
    import yaml
except ImportError:
    yaml = None


def load_class_names(data_dir: Path):
    yaml_path = data_dir / "data.yaml"
    if yaml_path.exists() and yaml is not None:
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
            names = cfg.get("names")
            if isinstance(names, dict):
                return {int(k): v for k, v in names.items()}
            if isinstance(names, list):
                return {i: n for i, n in enumerate(names)}
    return {}


def find_splits(data_dir: Path):
    """Find train/valid/test subfolders that contain images/ and labels/."""
    splits = []
    for split in ["train", "valid", "val", "test"]:
        img_dir = data_dir / split / "images"
        lbl_dir = data_dir / split / "labels"
        if img_dir.exists() and lbl_dir.exists():
            splits.append((split, img_dir, lbl_dir))
    if not splits:
        raise SystemExit(
            f"Couldn't find train/images + train/labels style folders under {data_dir}. "
            "Point --data at the folder that contains train/valid/test subfolders."
        )
    return splits


def analyze(data_dir: Path, imgsz: int, min_px: float, out_csv: Path | None):
    class_names = load_class_names(data_dir)
    splits = find_splits(data_dir)

    # per-class list of (scaled_px_width, scaled_px_height, split, filename)
    stats = defaultdict(list)
    total_boxes = 0
    missing_images = 0

    for split_name, img_dir, lbl_dir in splits:
        label_files = list(lbl_dir.glob("*.txt"))
        for lbl_path in label_files:
            # find matching image (try common extensions)
            img_path = None
            for ext in [".jpg", ".jpeg", ".png", ".JPG", ".PNG"]:
                candidate = img_dir / (lbl_path.stem + ext)
                if candidate.exists():
                    img_path = candidate
                    break
            if img_path is None:
                missing_images += 1
                continue

            with Image.open(img_path) as im:
                img_w, img_h = im.size

            # YOLO resizes preserving aspect ratio to fit imgsz, then pads.
            # The effective scale factor applied to the longer side:
            scale = imgsz / max(img_w, img_h)

            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls_id = int(float(parts[0]))
                    w_norm, h_norm = float(parts[3]), float(parts[4])

                    raw_px_w = w_norm * img_w
                    raw_px_h = h_norm * img_h

                    # size the box will actually have once resized for training
                    scaled_px_w = raw_px_w * scale
                    scaled_px_h = raw_px_h * scale

                    stats[cls_id].append((scaled_px_w, scaled_px_h, split_name, lbl_path.name))
                    total_boxes += 1

    if missing_images:
        print(f"Warning: {missing_images} label files had no matching image file, skipped.\n")

    print(f"Analyzed {total_boxes} boxes across {len(splits)} split(s), "
          f"scaled to imgsz={imgsz}\n")

    rows_for_csv = []
    print(f"{'class':<20}{'count':<8}{'mean_px':<10}{'median_px':<12}{'min_px':<9}{'%below_min':<12}")
    print("-" * 75)

    for cls_id in sorted(stats.keys()):
        boxes = stats[cls_id]
        # use the smaller of width/height per box as the limiting dimension
        sizes = sorted(min(w, h) for w, h, _, _ in boxes)
        n = len(sizes)
        mean_px = sum(sizes) / n
        median_px = sizes[n // 2]
        min_px_val = sizes[0]
        n_below = sum(1 for s in sizes if s < min_px)
        pct_below = 100 * n_below / n

        name = class_names.get(cls_id, str(cls_id))
        print(f"{name:<20}{n:<8}{mean_px:<10.1f}{median_px:<12.1f}{min_px_val:<9.1f}{pct_below:<12.1f}")

        for w, h, split_name, fname in boxes:
            rows_for_csv.append({
                "class": name,
                "split": split_name,
                "file": fname,
                "scaled_width_px": round(w, 1),
                "scaled_height_px": round(h, 1),
                "below_threshold": min(w, h) < min_px,
            })

    print(f"\nThreshold: boxes with the smaller side under {min_px}px at imgsz={imgsz} "
          f"are flagged as hard-to-detect.")
    print("If a large fraction of a class falls below this, that class needs closer-range "
          "images, tiling/slicing at inference (e.g. SAHI), or a larger imgsz.")

    if out_csv:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_for_csv[0].keys()))
            writer.writeheader()
            writer.writerows(rows_for_csv)
        print(f"\nPer-box detail written to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to dataset root (contains train/valid/test)")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size (default 640)")
    parser.add_argument("--min-px", type=float, default=25, help="Pixel threshold to flag as too small")
    parser.add_argument("--csv", default=None, help="Optional path to write per-box CSV detail")
    args = parser.parse_args()

    analyze(
        data_dir=Path(args.data),
        imgsz=args.imgsz,
        min_px=args.min_px,
        out_csv=Path(args.csv) if args.csv else None,
    )