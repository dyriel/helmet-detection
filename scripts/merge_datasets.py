"""
merge_datasets.py

Merges the cleaned Roboflow CCTV dataset and the converted Myanmar dataset
into ONE training set with two classes:

    0 with_helmet   (a motorcycle whose riders all wear helmets)
    1 no_helmet     (a motorcycle where at least one rider has no helmet)

Both datasets box the whole motorcycle with its riders, so they combine directly.

How the original classes are mapped (by NAME, read from each data.yaml):
    Roboflow:  rider_full_face, rider_half_face -> with_helmet
               rider_no_helmet                  -> no_helmet
               rider_helmet_invalid             -> your choice via --invalid
    Myanmar:   moto_with_helmet -> with_helmet
               moto_no_helmet   -> no_helmet

Splits are kept as they are (train stays train, valid stays valid, test stays
test), so nothing crosses between splits. Inputs are never modified; the merged
set is written to a new folder. Images are prefixed rf_ / mm_ so the source is
always visible in the file name, and manifest.csv records source and original
class names for every image.

Usage (run from the scripts folder):
    python merge_datasets.py --roboflow ..\\dataset --myanmar ..\\dataset_myanmar --out ..\\dataset_merged
    (add  --closeup ..\\dataset_closeup  to include close-range photos)

Choose what happens to rider_helmet_invalid with --invalid:
    no_helmet   (default) count it as a violation
    with_helmet           count it as compliant
    exclude               leave out every image that contains one
"""

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

OUT_NAMES = ["with_helmet", "no_helmet"]
RF_FALLBACK = ["rider_full_face", "rider_half_face", "rider_helmet_invalid", "rider_no_helmet"]
SPLITS = {"train": ["train"], "valid": ["valid", "val"], "test": ["test"]}
IMG_EXT = {".jpg", ".jpeg", ".png"}
EXCLUDE = "EXCLUDE"


def read_names(ds: Path, fallback=None):
    """Return {class_id: name} from the dataset's data.yaml."""
    p = ds / "data.yaml"
    if p.exists() and yaml is not None:
        with open(p) as f:
            names = (yaml.safe_load(f) or {}).get("names")
        if isinstance(names, dict):
            return {int(k): v for k, v in names.items()}
        if isinstance(names, list):
            return dict(enumerate(names))
    if fallback:
        print(f"WARNING: no usable data.yaml in {ds}; assuming class order {fallback}")
        return dict(enumerate(fallback))
    raise SystemExit(f"Can't read class names: {p} is missing (and pyyaml may not be installed).")


def rf_target(name, invalid):
    if name in ("rider_full_face", "rider_half_face"):
        return 0
    if name == "rider_no_helmet":
        return 1
    if name == "rider_helmet_invalid":
        return {"no_helmet": 1, "with_helmet": 0, "exclude": EXCLUDE}[invalid]
    return None


def cu_target(name, invalid):
    return {"with_helmet": 0, "no_helmet": 1}.get(name)


def mm_target(name, invalid):
    return {"moto_with_helmet": 0, "moto_no_helmet": 1}.get(name)


def process(ds, source, prefix, target_fn, invalid, out, writer, stats, problems, fallback=None):
    names = read_names(ds, fallback)
    for split, dirs in SPLITS.items():
        for d in dirs:
            img_dir, lbl_dir = ds / d / "images", ds / d / "labels"
            if not img_dir.exists():
                continue
            (out / split / "images").mkdir(parents=True, exist_ok=True)
            (out / split / "labels").mkdir(parents=True, exist_ok=True)
            for img in sorted(img_dir.iterdir()):
                if img.suffix.lower() not in IMG_EXT:
                    continue
                lbl = lbl_dir / (img.stem + ".txt")
                rows, orig, exclude = [], [], False
                if lbl.exists():
                    for line in lbl.read_text().splitlines():
                        parts = line.split()
                        if not parts:
                            continue
                        if len(parts) != 5:
                            problems[f"{source}: malformed label line (not 5 fields)"] += 1
                            continue
                        name = names.get(int(float(parts[0])))
                        target = target_fn(name, invalid)
                        if target == EXCLUDE:
                            exclude = True
                            break
                        if target is None:
                            problems[f"{source}: unmapped class '{name}' (boxes skipped)"] += 1
                            continue
                        rows.append(f"{target} " + " ".join(parts[1:5]))
                        orig.append(name)
                else:
                    problems[f"{source}: image with no label file (kept as empty background)"] += 1
                if exclude:
                    stats[(split, source, "excluded_images")] += 1
                    continue
                new_name = f"{prefix}{img.name}"
                shutil.copy2(img, out / split / "images" / new_name)
                (out / split / "labels" / f"{prefix}{img.stem}.txt").write_text(
                    "\n".join(rows) + ("\n" if rows else ""))
                writer.writerow([new_name, source, split, "|".join(orig)])
                stats[(split, source, "images")] += 1
                for r in rows:
                    stats[(split, source, "no_helmet" if r.startswith("1 ") else "with_helmet")] += 1


def main(rf: Path, mm: Path, out: Path, invalid: str, cu: Path = None):
    checks = [(rf, "Roboflow"), (mm, "Myanmar")] + ([(cu, "Close-up")] if cu else [])
    for p, label in checks:
        if not p.exists():
            raise SystemExit(f"{label} folder not found: {p}")
    if (out / "train").exists():
        raise SystemExit(f"{out} already has a train folder. Delete {out} first so old and new files don't mix.")
    out.mkdir(parents=True, exist_ok=True)

    stats, problems = Counter(), Counter()
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "source", "split", "original_classes"])
        process(rf, "roboflow_cctv", "rf_", rf_target, invalid, out, w, stats, problems, RF_FALLBACK)
        process(mm, "myanmar_helmet", "mm_", mm_target, invalid, out, w, stats, problems)
        if cu:
            process(cu, "closeup", "cu_", cu_target, invalid, out, w, stats, problems)

    lines = [f"path: {out.resolve().as_posix()}", "train: train/images", "val: valid/images",
             "test: test/images", "names:"] + [f"  {i}: {n}" for i, n in enumerate(OUT_NAMES)]
    (out / "data.yaml").write_text("\n".join(lines) + "\n")

    print(f"\nMerged dataset written to {out}   (rider_helmet_invalid -> {invalid})\n")
    print(f"{'split':<7}{'source':<16}{'images':>8}{'with_helmet':>13}{'no_helmet':>11}{'%no_helmet':>12}")
    for split in SPLITS:
        for src in ("roboflow_cctv", "myanmar_helmet") + (("closeup",) if cu else ()):
            n = stats[(split, src, "images")]
            a, b = stats[(split, src, "with_helmet")], stats[(split, src, "no_helmet")]
            pct = f"{100 * b / (a + b):.0f}%" if a + b else "-"
            ex = stats[(split, src, "excluded_images")]
            note = f"   ({ex} images excluded)" if ex else ""
            print(f"{split:<7}{src:<16}{n:>8}{a:>13}{b:>11}{pct:>12}{note}")
    tr_rf, tr_mm = stats[("train", "roboflow_cctv", "images")], stats[("train", "myanmar_helmet", "images")]
    if tr_rf + tr_mm:
        share = 100 * tr_mm / (tr_rf + tr_mm)
        print(f"\nTrain images: {share:.0f}% Myanmar / {100 - share:.0f}% Roboflow.")
        if share > 75 or share < 25:
            print("That is lopsided; the larger source will dominate what the model learns.")
    if problems:
        print("\nThings to know:")
        for k, v in problems.items():
            print(f"  {k}: {v}")
    print(f"\nTrain with:  yolo detect train data={out / 'data.yaml'} model=yolov8n.pt imgsz=960")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--roboflow", required=True, help="Cleaned Roboflow dataset (e.g. ..\\dataset)")
    p.add_argument("--myanmar", required=True, help="Converted Myanmar dataset (e.g. ..\\dataset_myanmar)")
    p.add_argument("--out", required=True, help="New folder for the merged dataset")
    p.add_argument("--invalid", choices=["no_helmet", "with_helmet", "exclude"], default="no_helmet",
                   help="What to do with rider_helmet_invalid (default: no_helmet)")
    p.add_argument("--closeup", default=None,
                   help="Optional: close-range dataset made by autolabel_closeup.py (e.g. ..\\dataset_closeup)")
    a = p.parse_args()
    main(Path(a.roboflow), Path(a.myanmar), Path(a.out), a.invalid, Path(a.closeup) if a.closeup else None)