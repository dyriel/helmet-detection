"""
check_split_leakage.py

Checks whether the same source frame (before Roboflow's hash suffix) appears
in both the train split and the valid/test splits. If train images are
augmented copies of frames that also appear raw in valid/test, that's a
data leak -- the model would effectively be evaluated on near-duplicates
of what it trained on.

Roboflow filenames look like: <source_frame>_jpg.rf.<hash>.jpg
This script strips everything from "_jpg.rf." onward to get the source
frame identifier, then compares sets across splits.

Usage:
    python check_split_leakage.py --data ..\\dataset
"""

import argparse
import re
from pathlib import Path
from collections import defaultdict


def source_key(filename: str) -> str:
    """Strip the Roboflow hash suffix to get the underlying source frame id."""
    # matches e.g. "20220802_075652_mp4-49_jpg.rf.d74ade15a50a3b3.jpg"
    # -> "20220802_075652_mp4-49"
    stem = Path(filename).stem  # drops extension
    match = re.match(r"^(.*)_jpg\.rf\.", stem)
    if match:
        return match.group(1)
    # fallback: if pattern doesn't match, just use the full stem
    return stem


def clip_key(filename: str) -> str:
    """
    Strip both the hash suffix AND the frame number to get the underlying
    video clip id. E.g. "20220802_075652_mp4-49_jpg.rf.<hash>.jpg"
    -> "20220802_075652_mp4"
    This catches near-duplicate leakage: different frame numbers from the
    SAME clip (a few frames apart) are visually near-identical even though
    source_key() would treat them as different images.
    """
    src = source_key(filename)
    match = re.match(r"^(.*)-\d+$", src)
    if match:
        return match.group(1)
    return src  # fallback if the -<framenum> pattern isn't present


def collect_keys(img_dir: Path):
    keys = defaultdict(list)
    clip_keys = defaultdict(list)
    for f in img_dir.glob("*.*"):
        keys[source_key(f.name)].append(f.name)
        clip_keys[clip_key(f.name)].append(f.name)
    return keys, clip_keys


def main(data_dir: Path):
    frame_splits = {}
    clip_splits = {}
    for split in ["train", "valid", "val", "test"]:
        img_dir = data_dir / split / "images"
        if img_dir.exists():
            frame_keys, clip_keys = collect_keys(img_dir)
            frame_splits[split] = frame_keys
            clip_splits[split] = clip_keys
            print(f"{split}: {len(frame_keys)} unique frames, "
                  f"{len(clip_keys)} unique clips, "
                  f"{sum(len(v) for v in frame_keys.values())} total images")

    if "train" not in frame_splits:
        print("No train/images folder found.")
        return

    def check_overlap(splits_dict, label):
        keys_a = set(splits_dict["train"].keys())
        found_any = False
        for split in ["valid", "val", "test"]:
            if split not in splits_dict:
                continue
            keys_b = set(splits_dict[split].keys())
            overlap = keys_a & keys_b
            if overlap:
                found_any = True
                print(f"\n⚠ {label} LEAK: {len(overlap)} {label.lower()}s appear in BOTH train and {split}:")
                for k in list(overlap)[:10]:
                    print(f"    {k}")
                if len(overlap) > 10:
                    print(f"    ... and {len(overlap) - 10} more")
            else:
                print(f"\n✓ No {label.lower()} overlap between train and {split}")
        return found_any

    print("\n" + "=" * 60)
    print("EXACT FRAME check (same file re-used across splits):")
    print("=" * 60)
    exact_leak = check_overlap(frame_splits, "Exact-frame")

    print("\n" + "=" * 60)
    print("CLIP-LEVEL check (near-duplicate frames from the same video, a few")
    print("frames apart -- this catches leakage the exact-frame check misses):")
    print("=" * 60)
    clip_leak = check_overlap(clip_splits, "Clip")

    print("\n" + "=" * 60)
    if clip_leak:
        print("Result: CLIP-LEVEL leakage detected. Even though exact frames differ,")
        print("train and valid/test share footage from the same video clips -- those")
        print("frames are likely near-identical. This dataset's split needs to be")
        print("redone at the clip level (see recommendation).")
    elif exact_leak:
        print("Result: exact-frame leakage detected.")
    else:
        print("Result: splits are clean at both the frame and clip level.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to dataset root")
    args = parser.parse_args()
    main(Path(args.data))