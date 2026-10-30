"""
visualize_boxes.py

Draws YOLO-format bounding boxes onto sample images so you can visually
check what's actually being annotated (tight head/helmet box? full body?
is there a separate motorcycle box?).

Usage:
    python visualize_boxes.py --data ..\\dataset --split train --n 12

Saves annotated images into a new "preview\\" folder next to the script
so you can open them and look, rather than displaying in a popup window.
"""

import argparse
import random
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
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


COLORS = [
    (255, 0, 0), (0, 200, 0), (0, 100, 255), (255, 165, 0),
    (200, 0, 200), (0, 200, 200), (150, 75, 0), (255, 255, 0),
]


def draw_boxes(img_path: Path, lbl_path: Path, class_names: dict, out_path: Path):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    if not lbl_path.exists():
        img.save(out_path)
        return

    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:])

            x1 = (xc - bw / 2) * w
            y1 = (yc - bh / 2) * h
            x2 = (xc + bw / 2) * w
            y2 = (yc + bh / 2) * h

            color = COLORS[cls_id % len(COLORS)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            label = class_names.get(cls_id, str(cls_id))
            draw.rectangle([x1, y1, x1 + 8 * len(label) + 6, y1 + 16], fill=color)
            draw.text((x1 + 3, y1 + 2), label, fill=(255, 255, 255))

    img.save(out_path)


def main(data_dir: Path, split: str, n: int, out_dir: Path, target_file: str = None, class_filter: str = None):
    class_names = load_class_names(data_dir)
    img_dir = data_dir / split / "images"
    lbl_dir = data_dir / split / "labels"

    if not img_dir.exists():
        raise SystemExit(f"No images found at {img_dir}")

    out_dir.mkdir(exist_ok=True)

    if target_file:
        # user wants one specific image, not a random sample
        img_path = img_dir / target_file
        if not img_path.exists():
            # try matching by stem in case extension differs
            matches = list(img_dir.glob(Path(target_file).stem + ".*"))
            if not matches:
                raise SystemExit(f"Couldn't find {target_file} in {img_dir}")
            img_path = matches[0]
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        out_path = out_dir / img_path.name
        draw_boxes(img_path, lbl_path, class_names, out_path)
        print(f"Saved: {out_path}")
        print(f"Open that file to see the box(es) drawn on it.")
        return

    all_images = list(img_dir.glob("*.*"))
    if not all_images:
        raise SystemExit(f"No image files in {img_dir}")

    if class_filter:
        # find the class id matching the given name
        name_to_id = {v: k for k, v in class_names.items()}
        if class_filter not in name_to_id:
            raise SystemExit(f"Class '{class_filter}' not found. Available: {list(name_to_id.keys())}")
        target_id = name_to_id[class_filter]

        matching_images = []
        for img_path in all_images:
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts and int(float(parts[0])) == target_id:
                        matching_images.append(img_path)
                        break
        print(f"Found {len(matching_images)} images containing class '{class_filter}'")
        all_images = matching_images
        if not all_images:
            raise SystemExit("No images found with that class.")

    sample = random.sample(all_images, min(n, len(all_images)))

    for img_path in sample:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        out_path = out_dir / img_path.name
        draw_boxes(img_path, lbl_path, class_names, out_path)
        print(f"Saved: {out_path}")

    print(f"\n{len(sample)} annotated preview images saved to: {out_dir}")
    print("Open that folder and look through them.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to dataset root")
    parser.add_argument("--split", default="train", choices=["train", "valid", "val", "test"])
    parser.add_argument("--n", type=int, default=12, help="Number of random images to preview")
    parser.add_argument("--file", default=None, help="Preview one specific image filename instead of a random sample")
    parser.add_argument("--class-name", default=None, help="Only sample images that contain this class (e.g. rider_helmet_invalid)")
    parser.add_argument("--out", default="preview", help="Output folder for annotated images")
    args = parser.parse_args()

    main(
        data_dir=Path(args.data),
        split=args.split,
        n=args.n,
        out_dir=Path(args.out),
        target_file=args.file,
        class_filter=args.class_name,
    )