"""
collect_reviewed_images.py

Run this AFTER you've manually gone through the review folders created by
group_duplicates_for_review.py and deleted the copies you don't want.

Rebuilds a clean train/images + train/labels folder using:
- Whatever images survived your manual review (from the review folder)
- The frames that never had duplicates in the first place (copied straight
  from the original dataset, since they never needed review)
- The matching label file for every kept image, pulled from the original
  dataset's labels folder

Usage:
    python collect_reviewed_images.py --data ..\\dataset --split train --review ..\\review --out ..\\dataset_clean
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


def main(data_dir: Path, split: str, review_dir: Path, out_dir: Path):
    orig_img_dir = data_dir / split / "images"
    orig_lbl_dir = data_dir / split / "labels"

    out_img_dir = out_dir / split / "images"
    out_lbl_dir = out_dir / split / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: figure out which frames had duplicates (were sent to review)
    by_frame = defaultdict(list)
    for img_path in orig_img_dir.glob("*.*"):
        by_frame[source_key(img_path.name)].append(img_path)
    reviewed_keys = {k for k, v in by_frame.items() if len(v) > 1}
    single_keys = {k for k, v in by_frame.items() if len(v) == 1}

    kept_count = 0
    empty_folders = []
    auto_resolved_count = 0

    # Step 2: pull from each review subfolder.
    # - If you already reviewed it (deleted down to 1 image), that survivor is used.
    # - If you HAVEN'T reviewed it yet (still has multiple images), auto-pick one
    #   instead of copying all of them -- this is what lets you stop reviewing
    #   partway through and let the rest be handled automatically.
    for key in reviewed_keys:
        safe_key = re.sub(r'[<>:"/\\|?*]', "_", key)
        frame_folder = review_dir / safe_key
        if not frame_folder.exists():
            print(f"  (warning: no review folder found for {key}, skipping)")
            continue
        survivors = sorted(frame_folder.glob("*.*"))
        if not survivors:
            empty_folders.append(key)
            continue
        if len(survivors) > 1:
            # not manually reviewed yet -- auto-pick just the first one
            survivors = [survivors[0]]
            auto_resolved_count += 1
        for img_path in survivors:
            lbl_path = orig_lbl_dir / (img_path.stem + ".txt")
            shutil.copy2(img_path, out_img_dir / img_path.name)
            if lbl_path.exists():
                shutil.copy2(lbl_path, out_lbl_dir / (img_path.stem + ".txt"))
            else:
                print(f"  (warning: no label found for {img_path.name})")
            kept_count += 1

    # Step 3: carry over frames that never had duplicates, untouched
    for key in single_keys:
        img_path = by_frame[key][0]
        lbl_path = orig_lbl_dir / (img_path.stem + ".txt")
        shutil.copy2(img_path, out_img_dir / img_path.name)
        if lbl_path.exists():
            shutil.copy2(lbl_path, out_lbl_dir / (img_path.stem + ".txt"))
        kept_count += 1

    print(f"\nRebuilt {split} set: {kept_count} images written to {out_img_dir}")
    print(f"  - manually reviewed (single survivor found): {len(reviewed_keys) - auto_resolved_count - len(empty_folders)}")
    print(f"  - auto-resolved (not yet reviewed, picked automatically): {auto_resolved_count}")
    print(f"  - carried over untouched (never had duplicates): {len(single_keys)}")
    if empty_folders:
        print(f"\n{len(empty_folders)} review folders had ALL copies deleted "
              f"(no image kept for that frame):")
        for k in empty_folders[:10]:
            print(f"    {k}")
        if len(empty_folders) > 10:
            print(f"    ... and {len(empty_folders) - 10} more")
        print("If this wasn't intentional, that frame is now missing entirely.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to the ORIGINAL dataset root")
    parser.add_argument("--split", default="train", help="Which split was reviewed (default: train)")
    parser.add_argument("--review", required=True, help="Path to the review folder you cleaned up")
    parser.add_argument("--out", required=True, help="Path to write the cleaned dataset")
    args = parser.parse_args()

    main(Path(args.data), args.split, Path(args.review), Path(args.out))