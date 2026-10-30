"""
autolabel_closeup.py

Turns UNLABELED close-range photos into a YOLO dataset, without drawing any boxes by hand.

You do one simple thing: sort the photos into two folders.
    with_helmet folder : every rider on the motorcycle wears a helmet
    no_helmet folder   : at least one rider on the motorcycle has NO helmet
                         (bare head, cap, hat, hood, bandana ... anything that isn't a helmet)

The script then draws the boxes for you. It uses a standard pretrained detector (COCO
YOLOv8, which already knows "motorcycle" and "person") to find each motorcycle and the
people sitting on it, and makes ONE box around the motorcycle plus its riders. That is
the same kind of box the rest of your training data uses. The class (0 with_helmet /
1 no_helmet) comes from the folder you put the photo in.

Because the class comes from the folder, a photo with several motorcycles must have
the SAME answer for all of them. Photos that mix helmeted and un-helmeted bikes should
be left out (or cropped) - the script can't tell which bike is which.

Usage (run from the scripts folder):
    python autolabel_closeup.py --with-helmet ..\\closeup_raw\\with_helmet --no-helmet ..\\closeup_raw\\no_helmet --out ..\\dataset_closeup

Needs:  pip install ultralytics pillow pillow-heif
First run downloads the small COCO model (about 50 MB).

What you get in ..\\dataset_closeup:
    train\\ valid\\ test\\   images + labels (70 / 15 / 15 split, random but repeatable)
    data.yaml, manifest.csv
    _preview\\              every kept photo with its box drawn  <- LOOK AT THESE
    _skipped.txt           photos with no motorcycle found (left out)

Always flip through _preview before training. The automatic boxes are usually right,
but not always (a bystander counted as a rider, two bikes joined, a missed bike).
Delete bad photos from the dataset, or from your input folders and re-run.
"""

import argparse
import csv
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}
PERSON, MOTORCYCLE = 0, 3
NAMES = ["with_helmet", "no_helmet"]


def load_image(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter(a, b):
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def merge_duplicate_bikes(bikes, thr=0.7):
    """The detector sometimes boxes one motorcycle twice (whole bike + a part of it).
    If one box mostly sits inside another, treat them as one bike (their union)."""
    bikes = [tuple(b) for b in bikes]
    merged = True
    while merged:
        merged = False
        for i in range(len(bikes)):
            for j in range(i + 1, len(bikes)):
                a, b = bikes[i], bikes[j]
                if inter(a, b) / max(min(area(a), area(b)), 1e-6) >= thr:
                    bikes[i] = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
                    del bikes[j]
                    merged = True
                    break
            if merged:
                break
    return bikes


def group_boxes(bikes, people, min_overlap=0.2):
    """One box per motorcycle: the bike plus the people sitting on it.

    A person belongs to the bike they overlap most, provided the overlap is at least
    min_overlap of the person's own box AND the person's horizontal centre lies inside
    the bike's width (stops people standing next to a parked bike being counted).
    """
    members = {i: [] for i in range(len(bikes))}
    for p in people:
        best, best_ov = None, 0.0
        cx = (p[0] + p[2]) / 2
        for i, b in enumerate(bikes):
            pad = 0.15 * (b[2] - b[0])
            if not (b[0] - pad <= cx <= b[2] + pad):
                continue
            ov = inter(p, b)
            if ov / max(area(p), 1e-6) >= min_overlap and ov > best_ov:
                best, best_ov = i, ov
        if best is not None:
            members[best].append(p)
    out = []
    for i, b in enumerate(bikes):
        boxes = [b] + members[i]
        out.append((min(x[0] for x in boxes), min(x[1] for x in boxes),
                    max(x[2] for x in boxes), max(x[3] for x in boxes), len(members[i])))
    return out


def yolo_line(cls, box, w, h):
    x1, y1, x2, y2 = box[:4]
    return (f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
            f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")


def main(args):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("Missing dependency. Run: pip install ultralytics pillow pillow-heif")

    out = Path(args.out)
    if (out / "train").exists():
        raise SystemExit(f"{out} already has a train folder. Delete {out} first so old and new don't mix.")

    items = []
    for cls, folder in ((0, args.with_helmet), (1, args.no_helmet)):
        d = Path(folder)
        if not d.is_dir():
            raise SystemExit(f"Folder not found: {d}")
        files = sorted(f for f in d.iterdir() if f.suffix.lower() in IMG_EXT)
        print(f"{NAMES[cls]}: {len(files)} photos in {d}")
        items += [(f, cls) for f in files]
    if not items:
        raise SystemExit("No photos found.")

    det = YOLO(args.detector)
    kept, skipped = [], []
    for n, (f, cls) in enumerate(items, 1):
        try:
            img = load_image(f)
        except Exception as e:
            skipped.append(f"{f}  (can't open: {e})")
            continue
        r = det.predict(img, conf=args.conf, classes=[PERSON, MOTORCYCLE], imgsz=args.imgsz,
                        device=args.device, verbose=False)[0]
        xyxy = r.boxes.xyxy.cpu().numpy()
        clsid = r.boxes.cls.cpu().numpy().astype(int)
        bikes = [tuple(b) for b, c in zip(xyxy, clsid) if c == MOTORCYCLE]
        people = [tuple(b) for b, c in zip(xyxy, clsid) if c == PERSON]
        h, w = img.shape[:2]
        # ignore specks (far-away bikes would be mislabelled by the folder name)
        bikes = [b for b in bikes if area(b) >= args.min_area * w * h]
        if not bikes:
            skipped.append(f"{f}  (no motorcycle found)")
            continue
        bikes = merge_duplicate_bikes(bikes)
        if len(bikes) > args.max_bikes:
            skipped.append(f"{f}  (found {len(bikes)} motorcycles, more than --max-bikes {args.max_bikes}; "
                           "too busy to label from a folder name)")
            continue
        groups = group_boxes(bikes, people)
        kept.append((f, cls, img, groups))
        if n % 50 == 0:
            print(f"  {n}/{len(items)}", flush=True)

    if not kept:
        raise SystemExit("No motorcycles found in any photo.")

    rng = random.Random(args.seed)
    order = list(range(len(kept)))
    rng.shuffle(order)
    n_test = max(1, round(len(kept) * args.test_frac))
    n_valid = max(1, round(len(kept) * args.valid_frac))
    split_of = {}
    for rank, i in enumerate(order):
        split_of[i] = "test" if rank < n_test else "valid" if rank < n_test + n_valid else "train"

    for s in ("train", "valid", "test"):
        (out / s / "images").mkdir(parents=True)
        (out / s / "labels").mkdir(parents=True)
    (out / "_preview").mkdir()

    counts = {}
    with open(out / "manifest.csv", "w", newline="") as mf:
        w_ = csv.writer(mf)
        w_.writerow(["filename", "source", "split", "folder_class", "bikes", "riders_found"])
        for i, (f, cls, img, groups) in enumerate(kept):
            split = split_of[i]
            name = f"cu_{i:04d}_{f.stem}"[:80].replace(" ", "_")
            h, w = img.shape[:2]
            cv2.imwrite(str(out / split / "images" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (out / split / "labels" / f"{name}.txt").write_text(
                "\n".join(yolo_line(cls, g, w, h) for g in groups) + "\n")
            prev = img.copy()
            colour = (0, 200, 0) if cls == 0 else (0, 0, 255)
            t = max(2, round(max(h, w) / 400))
            for g in groups:
                cv2.rectangle(prev, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])), colour, t)
                cv2.putText(prev, f"{NAMES[cls]} ({g[4]} rider)", (int(g[0]) + 4, max(18, int(g[1]) + 22)),
                            cv2.FONT_HERSHEY_SIMPLEX, t / 3, colour, max(1, t - 1), cv2.LINE_AA)
            cv2.imwrite(str(out / "_preview" / f"{split}_{name}.jpg"), prev, [cv2.IMWRITE_JPEG_QUALITY, 85])
            w_.writerow([f"{name}.jpg", "closeup", split, NAMES[cls], len(groups), sum(g[4] for g in groups)])
            counts[(split, cls)] = counts.get((split, cls), 0) + len(groups)

    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
        "names:\n  0: with_helmet\n  1: no_helmet\n")
    if skipped:
        (out / "_skipped.txt").write_text("\n".join(skipped) + "\n")

    print(f"\nKept {len(kept)} photos, skipped {len(skipped)}"
          + (f" (listed in {out / '_skipped.txt'})" if skipped else "") + ".\n")
    print(f"{'split':<8}{'with_helmet boxes':>19}{'no_helmet boxes':>17}")
    for s in ("train", "valid", "test"):
        print(f"{s:<8}{counts.get((s, 0), 0):>19}{counts.get((s, 1), 0):>17}")
    print(f"\nNEXT: open {out / '_preview'} and check the boxes before using this data.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--with-helmet", required=True, help="folder of photos where every rider wears a helmet")
    p.add_argument("--no-helmet", required=True, help="folder of photos with at least one rider without a helmet")
    p.add_argument("--out", required=True, help="new dataset folder to create")
    p.add_argument("--detector", default="yolov8m.pt", help="COCO model used to find bikes and people")
    p.add_argument("--conf", type=float, default=0.3, help="detector confidence (default 0.3)")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--min-area", type=float, default=0.02,
                   help="ignore motorcycles smaller than this fraction of the photo (default 0.02)")
    p.add_argument("--max-bikes", type=int, default=3,
                   help="skip photos with more motorcycles than this (default 3); crowded scenes are "
                        "ambiguous because one folder label has to fit every bike")
    p.add_argument("--valid-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    main(p.parse_args())