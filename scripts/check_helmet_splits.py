"""
check_helmet_splits.py

Checks the Myanmar HELMET dataset's three CSVs (train / validation / test)
for clip-level leakage: the same video clip appearing in more than one split.

An image_name looks like "Bago_highway_11_01": clip = "Bago_highway_11",
frame = "01". Frames from the same clip are near-identical, so a clip must
live in exactly one split.

Usage (run from the scripts folder):
    python check_helmet_splits.py --train PATH\\helmet_detection_train.csv ^
        --valid PATH\\helmet_detection_validation.csv ^
        --test PATH\\helmet_detection_test.csv
"""

import argparse
import pandas as pd


def load_clips(csv_path):
    df = pd.read_csv(csv_path)
    # the validation CSV header may say image_id instead of image_name
    col = "image_name" if "image_name" in df.columns else "image_id"
    names = df[col].astype(str)
    clips = names.str.rsplit("_", n=1).str[0]
    return df, set(clips), names.nunique()


def main(train, valid, test):
    splits = {}
    for name, path in [("train", train), ("valid", valid), ("test", test)]:
        df, clips, n_frames = load_clips(path)
        splits[name] = clips
        print(f"{name}: {len(df)} rows, {n_frames} unique frames, {len(clips)} unique clips")

    print()
    leak = False
    for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]:
        overlap = splits[a] & splits[b]
        if overlap:
            leak = True
            print(f"LEAK: {len(overlap)} clips appear in both {a} and {b}, e.g. {sorted(overlap)[:5]}")
        else:
            print(f"OK: no clips shared between {a} and {b}")

    print()
    if leak:
        print("Result: clip-level leakage found. We'd re-split by clip before training.")
    else:
        print("Result: the CSV splits are clean at the clip level.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", required=True)
    p.add_argument("--valid", required=True)
    p.add_argument("--test", required=True)
    a = p.parse_args()
    main(a.train, a.valid, a.test)