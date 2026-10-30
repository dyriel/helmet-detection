"""
inspect_helmet_csv.py

Inspects the HELMET dataset's flattened CSVs (helmet_detection_train.csv,
_test.csv, _validation.csv) and prints clean, structured info:
- all unique label values (the compound Driver/Passenger helmet strings)
- a handful of raw sample rows
- a sanity check that x2 > x1 and y2 > y1 for every row (confirms corner format)

Usage:
    python inspect_helmet_csv.py --csv helmet_detection_train.csv
"""

import argparse
import pandas as pd


def main(csv_path: str):
    df = pd.read_csv(csv_path)

    print(f"Loaded {csv_path}")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}\n")

    print("=" * 60)
    print("First 5 raw rows:")
    print("=" * 60)
    print(df.head(5).to_string())

    print("\n" + "=" * 60)
    print(f"All unique values in 'label' column ({df['label'].nunique()} total):")
    print("=" * 60)
    for val, count in df["label"].value_counts().items():
        pct = 100 * count / len(df)
        print(f"  {val:<40} count={count:<8} ({pct:.1f}%)")

    print("\n" + "=" * 60)
    print("Box coordinate sanity check:")
    print("=" * 60)
    bad_x = (df["x2"] <= df["x1"]).sum()
    bad_y = (df["y2"] <= df["y1"]).sum()
    print(f"Rows where x2 <= x1: {bad_x} ({'OK, format confirmed corner-to-corner' if bad_x == 0 else 'CHECK THIS - unexpected'})")
    print(f"Rows where y2 <= y1: {bad_y} ({'OK, format confirmed corner-to-corner' if bad_y == 0 else 'CHECK THIS - unexpected'})")

    print("\n" + "=" * 60)
    print("Sample of image_name values (to check naming pattern / frame numbering):")
    print("=" * 60)
    print(df["image_name"].head(10).to_list())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Path to the CSV file to inspect")
    args = parser.parse_args()
    main(args.csv)