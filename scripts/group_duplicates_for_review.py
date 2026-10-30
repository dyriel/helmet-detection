"""
group_duplicates_for_review.py

Finds frames that have multiple copies (e.g. raw + augmented versions from
Roboflow) and groups each frame's copies into their own subfolder, so you
can open each subfolder, look at the images side by side, delete the ones
you don't want, and keep the rest.

Frames with only ONE copy (no duplicates) are left alone -- there's nothing
to review for those, so they're reported but not moved.

Usage:
    python group_duplicates_for_review.py --data ..\\dataset --split train --out ..\\review

After this, go through each subfolder in ..\\review, open the images, and
DELETE the copies you don't want to keep (leave at least one per folder).
Then run collect_reviewed_images.py to rebuild a clean train folder from
whatever's left.
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


def main(data_dir: Path, split: str, out_dir: Path):
    img_dir = data_dir / split / "images"
    if not img_dir.exists():
        raise SystemExit(f"No folder found at {img_dir}")

    by_frame = defaultdict(list)
    for img_path in img_dir.glob("*.*"):
        by_frame[source_key(img_path.name)].append(img_path)

    duplicates = {k: v for k, v in by_frame.items() if len(v) > 1}
    singles = {k: v for k, v in by_frame.items() if len(v) == 1}

    print(f"Total unique frames: {len(by_frame)}")
    print(f"Frames with duplicates (need review): {len(duplicates)}")
    print(f"Frames with only one copy (no review needed): {len(singles)}\n")

    out_dir.mkdir(parents=True, exist_ok=True)

    for key, paths in duplicates.items():
        # sanitize key for use as a folder name (in case it has odd characters)
        safe_key = re.sub(r'[<>:"/\\|?*]', "_", key)
        frame_folder = out_dir / safe_key
        frame_folder.mkdir(parents=True, exist_ok=True)
        for p in paths:
            shutil.copy2(p, frame_folder / p.name)

    print(f"Copied {len(duplicates)} frame-groups into: {out_dir}")
    print("Each subfolder contains that frame's copies (raw + augmented, or")
    print("multiple augmented variants). Open each one, look at the images,")
    print("and DELETE the ones you don't want -- keep at least one per folder.")
    print("\nWhen done, run collect_reviewed_images.py to rebuild the train set.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to dataset root")
    parser.add_argument("--split", default="train", help="Which split to review (default: train)")
    parser.add_argument("--out", required=True, help="Folder to create the per-frame review subfolders in")
    args = parser.parse_args()

    main(Path(args.data), args.split, Path(args.out))