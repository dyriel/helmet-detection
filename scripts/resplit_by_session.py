"""
resplit_by_session.py

Fixes a dataset where images were split randomly by frame (causing session
leakage across train/valid/test) by re-splitting at the SESSION level instead.

What it does:
1. Pools all images+labels from the existing train/valid/test folders.
2. Deduplicates to one image per unique frame (drops extra augmented copies
   of the same source frame -- see note in the chat about why).
3. Assigns entire sessions to train (you choose which).
4. Splits the one remaining held-out session's frames chronologically into
   valid and test (first chunk -> valid, last chunk -> test), so they aren't
   interleaved frame-by-frame.
5. Copies images + matching label files into a new clean folder structure.

Usage:
    python resplit_by_session.py --data ..\\dataset --out ..\\dataset_resplit ^
        --train-sessions "New-video1_mp4,New-video2_mp4" ^
        --heldout-session "20220802_075652_mp4" ^
        --valid-ratio 0.6
"""

import argparse
import re
import shutil
from pathlib import Path
from collections import defaultdict


def source_key(filename: str) -> str:
    stem = Path(filename).stem
    match = re.match(r"^(.*)_jpg\.rf\.", stem)
    return match.group(1) if match else stem


def clip_key(filename: str) -> str:
    src = source_key(filename)
    match = re.match(r"^(.*)-(\d+)$", src)
    if match:
        return match.group(1), int(match.group(2))
    return src, 0


def collect_all_images(data_dir: Path):
    """Pool images from all existing splits, keyed by unique source frame."""
    by_frame = {}  # source_key -> (image_path, label_path)
    for split in ["train", "valid", "val", "test"]:
        img_dir = data_dir / split / "images"
        lbl_dir = data_dir / split / "labels"
        if not img_dir.exists():
            continue
        for img_path in img_dir.glob("*.*"):
            key = source_key(img_path.name)
            if key in by_frame:
                continue  # dedup: keep the first copy we saw, skip extra augmented dupes
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            by_frame[key] = (img_path, lbl_path if lbl_path.exists() else None)
    return by_frame


def copy_pair(img_path: Path, lbl_path: Path, out_dir: Path, split: str):
    img_out = out_dir / split / "images"
    lbl_out = out_dir / split / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_path, img_out / img_path.name)
    if lbl_path is not None:
        shutil.copy2(lbl_path, lbl_out / (img_path.stem + ".txt"))
    else:
        print(f"  (warning: no label file found for {img_path.name})")


def main(data_dir: Path, out_dir: Path, train_sessions, heldout_session, valid_ratio):
    by_frame = collect_all_images(data_dir)
    print(f"Pooled {len(by_frame)} unique frames after dedup.\n")

    # group by session
    sessions = defaultdict(list)  # session_name -> list of (frame_num, source_key)
    for key, (img_path, lbl_path) in by_frame.items():
        session, frame_num = clip_key(img_path.name)
        sessions[session].append((frame_num, key, img_path, lbl_path))

    print("Sessions found:")
    for s, items in sessions.items():
        print(f"  {s}: {len(items)} frames")
    print()

    missing = [s for s in train_sessions + [heldout_session] if s not in sessions]
    if missing:
        raise SystemExit(f"These session names weren't found in the data: {missing}\n"
                          f"Available sessions: {list(sessions.keys())}")

    train_count = 0
    for s in train_sessions:
        for frame_num, key, img_path, lbl_path in sessions[s]:
            copy_pair(img_path, lbl_path, out_dir, "train")
            train_count += 1
    print(f"train: {train_count} images from sessions {train_sessions}")

    held = sorted(sessions[heldout_session], key=lambda x: x[0])  # sort by frame number
    split_point = int(len(held) * valid_ratio)
    valid_items = held[:split_point]
    test_items = held[split_point:]

    for frame_num, key, img_path, lbl_path in valid_items:
        copy_pair(img_path, lbl_path, out_dir, "valid")
    for frame_num, key, img_path, lbl_path in test_items:
        copy_pair(img_path, lbl_path, out_dir, "test")

    print(f"valid: {len(valid_items)} images from session {heldout_session} (earlier frames)")
    print(f"test:  {len(test_items)} images from session {heldout_session} (later frames)")

    print(f"\nDone. New split written to: {out_dir}")
    print("Remember to copy/update data.yaml into the new folder with the correct")
    print("train/valid/test paths and your class list before training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to the current (leaky) dataset root")
    parser.add_argument("--out", required=True, help="Path to write the re-split dataset")
    parser.add_argument("--train-sessions", required=True,
                         help="Comma-separated session names to use fully for training")
    parser.add_argument("--heldout-session", required=True,
                         help="Session name to split into valid/test")
    parser.add_argument("--valid-ratio", type=float, default=0.6,
                         help="Fraction of the held-out session's frames used for valid (rest goes to test)")
    args = parser.parse_args()

    main(
        out_dir=Path(args.out),
        train_sessions=[s.strip() for s in args.train_sessions.split(",")],
        heldout_session=args.heldout_session.strip(),
        valid_ratio=args.valid_ratio,
    )