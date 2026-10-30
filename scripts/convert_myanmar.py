"""
convert_myanmar.py

Converts one extracted split of the Myanmar HELMET dataset (v2 format) into
YOLO training format.

Input layout (what you have):
    raw_myanmar\\test_v2\\<clip>\\images\\01.jpg ... 100.jpg
    raw_myanmar\\test_v2\\<clip>\\annotations\\01.txt ... 100.txt
Each annotation line:  <label> <track_id> <x_center> <y_center> <w> <h>
(boxes are already normalised 0-1; the box covers the whole motorcycle)

What it does:
  * Decodes the compound label (e.g. DHelmetP1NoHelmet): if ANY rider on the
    motorcycle has no helmet -> moto_no_helmet, otherwise moto_with_helmet.
  * Keeps every Nth frame per clip (default 5). Neighbouring frames of a 10 fps
    clip are near-identical, so keeping all of them adds almost nothing.
  * Skips frames with no motorcycle boxes (use --keep-empty to keep them).
  * Names outputs <clip>_<frame>.jpg so they stay traceable to the source.
  * Writes a manifest.csv (source, clip, frame, split, original label).
  * Never touches the raw folder.

Usage (run from the scripts folder), once per downloaded split:
    python convert_myanmar.py --raw ..\\raw_myanmar\\test_v2 --out ..\\dataset_myanmar --split test
    python convert_myanmar.py --raw ..\\raw_myanmar\\validation_v2 --out ..\\dataset_myanmar --split valid
    python convert_myanmar.py --raw ..\\raw_myanmar\\training_v2 --out ..\\dataset_myanmar --split train

Class ids follow the planned combined list (0,1 are the CCTV head classes and
are unused here):  0 head_with_helmet, 1 head_no_helmet,
                   2 moto_with_helmet, 3 moto_no_helmet
"""

import argparse
import csv
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

NAMES = ["head_with_helmet", "head_no_helmet", "moto_with_helmet", "moto_no_helmet"]
TOKEN = re.compile(r"(D|P[0-3])(NoHelmet|Helmet)")


def decode(label: str):
    """'DHelmetP1NoHelmet' -> ('moto_no_helmet'), or None if unparseable."""
    tokens = TOKEN.findall(label)
    # the tokens must account for the whole string, or it's something unexpected
    if not tokens or "".join(p + s for p, s in tokens) != label:
        return None
    return "moto_no_helmet" if any(s == "NoHelmet" for _, s in tokens) else "moto_with_helmet"


def write_yaml(out: Path):
    yaml_path = out / "data.yaml"
    if yaml_path.exists():
        return
    lines = ["path: .", "train: train/images", "val: valid/images", "test: test/images", "names:"]
    lines += [f"  {i}: {n}" for i, n in enumerate(NAMES)]
    yaml_path.write_text("\n".join(lines) + "\n")


def site_of(clip_name: str) -> str:
    """'Bago_highway_10' -> 'Bago_highway', 'Mandalay_1_128' -> 'Mandalay_1'."""
    return re.sub(r"_\d+$", "", clip_name)


def clip_number(clip_name: str) -> int:
    m = re.search(r"_(\d+)$", clip_name)
    return int(m.group(1)) if m else 0


def clip_brightness(clip: Path) -> float:
    """Rough lighting level of a clip: mean grey of its middle frame, 0-255."""
    try:
        from PIL import Image
    except ImportError:
        raise SystemExit("--stratify-brightness needs pillow: pip install pillow --break-system-packages")
    frames = sorted((clip / "images").glob("*.jpg"))
    if not frames:
        return 0.0
    with Image.open(frames[len(frames) // 2]) as im:
        im.draft("L", (240, 135))  # fast low-res decode
        small = im.convert("L").resize((32, 18))
        px = small.tobytes()  # one byte per pixel in greyscale mode
    return sum(px) / len(px)


def pick_by_brightness(clips, n):
    """Pick n clips spread evenly from darkest to brightest (stratified, no randomness)."""
    ranked = sorted((clip_brightness(c), c.name, c) for c in clips)
    if n == 1:
        return [ranked[len(ranked) // 2][2]]
    idxs = sorted({round(i * (len(ranked) - 1) / (n - 1)) for i in range(n)})
    return [ranked[i][2] for i in idxs]


def main(raw: Path, out: Path, split: str, every: int, keep_empty: bool, max_side: int,
         max_per_site: int, seed: int, list_sites: bool, stratify: bool, site_caps=None):
    site_caps = site_caps or {}
    if not raw.exists():
        raise SystemExit(f"Folder not found: {raw}")
    clips = sorted(p for p in raw.iterdir() if p.is_dir())
    if not clips:
        raise SystemExit(f"No clip folders inside {raw}")

    by_site = defaultdict(list)
    for c in clips:
        by_site[site_of(c.name)].append(c)

    print(f"Clips available per site ({len(clips)} total):")
    for s, lst in sorted(by_site.items()):
        print(f"  {s:<18} {len(lst)}")

    if list_sites:
        return

    if max_per_site or site_caps:
        unknown = [s for s in site_caps if s not in by_site]
        if unknown:
            raise SystemExit(f"--site-cap names site(s) that don't exist: {unknown}. "
                             f"Sites in this folder: {sorted(by_site)}")
        rng = random.Random(seed)
        chosen = []
        short = []
        for s, lst in sorted(by_site.items()):
            cap = site_caps.get(s, max_per_site)  # 0 means no limit for this site
            lst = sorted(lst, key=lambda c: clip_number(c.name))
            if not cap or len(lst) <= cap:
                picks = list(lst)
                if cap:
                    short.append(s)
            elif stratify:
                picks = pick_by_brightness(lst, cap)
            else:
                # split the site's clips (ordered by clip number) into `cap`
                # equal stretches and pick one random clip from each stretch, so the
                # picks are spread across the whole range instead of bunching up
                n, m = cap, len(lst)
                picks = [rng.choice(lst[k * m // n:(k + 1) * m // n]) for k in range(n)]
            chosen += picks
            print(f"  {s}: {len(picks)} clips (cap {cap or 'none'}), clip numbers {[clip_number(c.name) for c in picks]}")
        clips = sorted(chosen)
        how = ("spread from darkest to brightest" if stratify
               else f"spread across each site's clip numbers, seed={seed}")
        over = f"; overrides {site_caps}" if site_caps else ""
        print(f"\nSelected {len(clips)} clips (default cap {max_per_site or 'none'} per site{over}, {how}).")
        if short:
            print(f"Sites where every available clip was taken: {sorted(short)}")

    img_out = out / split / "images"
    lbl_out = out / split / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    write_yaml(out)

    resize = None
    if max_side:
        try:
            from PIL import Image
            resize = Image
        except ImportError:
            raise SystemExit("--max-side needs pillow: pip install pillow --break-system-packages")

    header = ["filename", "source", "clip", "frame", "split", "original_labels"]
    manifest_path = out / "manifest.csv"
    kept_rows = []  # rows from OTHER splits are kept; this split's old rows are replaced
    if manifest_path.exists():
        with open(manifest_path, newline="") as f:
            r = csv.reader(f)
            next(r, None)
            kept_rows = [row for row in r if len(row) > 4 and row[4] != split]
    existing = len(list(img_out.glob("*.jpg")))
    if existing:
        print(f"\nWARNING: {img_out} already holds {existing} images from an earlier run.")
        print("If you changed the options, delete that split's folder first, or old and new images will mix.")
    mf = open(manifest_path, "w", newline="")
    mw = csv.writer(mf)
    mw.writerow(header)
    mw.writerows(kept_rows)

    counts = Counter()
    bad_labels = Counter()
    site_stats = defaultdict(lambda: Counter())
    site_bikes = defaultdict(set)  # (site, class) -> distinct (clip, track_id)
    n_written = 0

    for ci, clip in enumerate(clips, 1):
        if ci % 10 == 0:
            print(f"  clip {ci}/{len(clips)} ... {n_written} images written so far", flush=True)
        img_dir, ann_dir = clip / "images", clip / "annotations"
        if not img_dir.exists() or not ann_dir.exists():
            counts["clips_missing_folders"] += 1
            continue
        frames = sorted(p.stem for p in img_dir.glob("*.jpg"))
        for idx, frame in enumerate(frames):
            if idx % every != 0:
                continue
            ann = ann_dir / f"{frame}.txt"
            rows, originals = [], []
            if ann.exists():
                for line in ann.read_text().splitlines():
                    parts = line.split()
                    if len(parts) != 6:
                        counts["malformed_lines"] += 1
                        continue
                    label, track, xc, yc, w, h = parts
                    cls_name = decode(label)
                    if cls_name is None:
                        bad_labels[label] += 1
                        continue
                    xc, yc, w, h = map(float, (xc, yc, w, h))
                    if w <= 0 or h <= 0 or not (0 <= xc <= 1 and 0 <= yc <= 1):
                        counts["bad_boxes"] += 1
                        continue
                    rows.append(f"{NAMES.index(cls_name)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                    originals.append(label)
                    counts[cls_name] += 1
                    site_stats[site_of(clip.name)][cls_name] += 1
                    site_bikes[(site_of(clip.name), cls_name)].add((clip.name, track))
            if not rows and not keep_empty:
                counts["frames_skipped_no_boxes"] += 1
                continue

            name = f"{clip.name}_{frame}"
            src_img = img_dir / f"{frame}.jpg"
            dst_img = img_out / f"{name}.jpg"
            if resize:
                with resize.open(src_img) as im:
                    scale = max_side / max(im.size)
                    if scale < 1:
                        im = im.resize((round(im.size[0] * scale), round(im.size[1] * scale)))
                    im.save(dst_img, quality=95)
            else:
                shutil.copy2(src_img, dst_img)
            (lbl_out / f"{name}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
            mw.writerow([f"{name}.jpg", "myanmar_helmet", clip.name, frame, split, "|".join(originals)])
            n_written += 1
            site_stats[site_of(clip.name)]["images"] += 1

    mf.close()
    (out / f"selected_clips_{split}.txt").write_text("\n".join(c.name for c in clips) + "\n")

    print("\n'boxes' count every kept frame; 'bikes' count each distinct motorcycle once (by track id),")
    print("which is the better measure of real variety, since one bike appears in many frames.\n")
    print(f"{'site':<18}{'images':>7}{'boxes_ok':>10}{'boxes_no':>10}{'bikes_ok':>10}{'bikes_no':>10}{'%no(bikes)':>12}")
    tot_ok = tot_no = 0
    for s, st in sorted(site_stats.items()):
        b_ok = len(site_bikes[(s, "moto_with_helmet")])
        b_no = len(site_bikes[(s, "moto_no_helmet")])
        tot_ok += b_ok
        tot_no += b_no
        pct = 100 * b_no / (b_ok + b_no) if (b_ok + b_no) else 0
        print(f"{s:<18}{st['images']:>7}{st['moto_with_helmet']:>10}{st['moto_no_helmet']:>10}"
              f"{b_ok:>10}{b_no:>10}{pct:>11.0f}%")
    print(f"{'ALL SITES':<18}{n_written:>7}{counts['moto_with_helmet']:>10}{counts['moto_no_helmet']:>10}"
          f"{tot_ok:>10}{tot_no:>10}")
    print()
    print(f"Clips processed: {len(clips)}")
    print(f"Images written to {img_out}: {n_written}")
    print(f"Boxes written: moto_with_helmet={counts['moto_with_helmet']}, "
          f"moto_no_helmet={counts['moto_no_helmet']}")
    print(f"Frames skipped (no boxes): {counts['frames_skipped_no_boxes']}")
    if counts["malformed_lines"] or counts["bad_boxes"] or counts["clips_missing_folders"]:
        print(f"Skipped: {counts['malformed_lines']} malformed lines, "
              f"{counts['bad_boxes']} invalid boxes, "
              f"{counts['clips_missing_folders']} clips missing images/annotations folders")
    if bad_labels:
        print("\nLabels the decoder did not recognise (these boxes were skipped):")
        for lab, n in bad_labels.most_common(10):
            print(f"  {lab}: {n}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", required=True, help="Extracted split folder, e.g. raw_myanmar\\test_v2")
    p.add_argument("--out", required=True, help="Output dataset folder, e.g. dataset_myanmar")
    p.add_argument("--split", required=True, choices=["train", "valid", "test"])
    p.add_argument("--every", type=int, default=5, help="Keep every Nth frame per clip (default 5)")
    p.add_argument("--keep-empty", action="store_true", help="Also keep frames with no motorcycle boxes")
    p.add_argument("--max-side", type=int, default=0,
                   help="Optional: shrink images so the longer side is at most this many px (e.g. 1280)")
    p.add_argument("--max-clips-per-site", type=int, default=0,
                   help="Pick at most this many clips from each site, spread out (repeatable). 0 = use all")
    p.add_argument("--seed", type=int, default=42, help="Seed for the clip choice (default 42)")
    p.add_argument("--list-sites", action="store_true",
                   help="Just print how many clips each site has, then exit (writes nothing)")
    p.add_argument("--stratify-brightness", action="store_true",
                   help="With --max-clips-per-site: spread the picks from the darkest to the brightest "
                        "clips at each site instead of by clip number, so different lighting is covered")
    p.add_argument("--site-cap", default="",
                   help="Per-site overrides of the cap, e.g. Mandalay_1=60,Yangon_II=27 "
                        "(0 means take every clip from that site)")
    a = p.parse_args()
    caps = {}
    for item in filter(None, (x.strip() for x in a.site_cap.split(","))):
        name, _, num = item.partition("=")
        if not num.strip().isdigit():
            raise SystemExit(f"Bad --site-cap entry '{item}'. Use Site=Number, e.g. Mandalay_1=60")
        caps[name.strip()] = int(num)
    main(Path(a.raw), Path(a.out), a.split, a.every, a.keep_empty, a.max_side,
         a.max_clips_per_site, a.seed, a.list_sites, a.stratify_brightness, caps)