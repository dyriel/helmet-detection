"""
shrink_and_zip.py

Makes a smaller copy of the merged dataset and zips it, ready to upload to
Google Drive / Colab.

  * Images whose longer side is above --max-side are shrunk to that size
    (default 1280). Smaller images are copied untouched.
  * Label files are copied as they are. They store positions as fractions of the
    image, so shrinking the picture does not change them.
  * The original dataset is never modified.

Usage (run from the scripts folder):
    python shrink_and_zip.py --src ..\\dataset_merged --out ..\\dataset_merged_small --max-side 1280

Produces the folder dataset_merged_small and the file dataset_merged_small.zip
next to it.
"""

import argparse
import shutil
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Missing dependency. Run: pip install pillow")

IMG_EXT = {".jpg", ".jpeg", ".png"}
SPLITS = ["train", "valid", "test"]


def main(src: Path, out: Path, max_side: int, quality: int):
    if not src.exists():
        raise SystemExit(f"Folder not found: {src}")
    if out.exists():
        raise SystemExit(f"{out} already exists. Delete it first so old and new files don't mix.")
    zip_path = out.parent / (out.name + ".zip")
    if zip_path.exists():
        raise SystemExit(f"{zip_path} already exists. Delete it first.")

    shrunk = copied = labels = 0
    for split in SPLITS:
        img_dir, lbl_dir = src / split / "images", src / split / "labels"
        if not img_dir.exists():
            continue
        (out / split / "images").mkdir(parents=True)
        (out / split / "labels").mkdir(parents=True)
        files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
        for i, img in enumerate(files, 1):
            dst = out / split / "images" / img.name
            with Image.open(img) as im:
                longest = max(im.size)
                if longest > max_side:
                    scale = max_side / longest
                    new = im.convert("RGB").resize(
                        (round(im.size[0] * scale), round(im.size[1] * scale)), Image.LANCZOS)
                    if img.suffix.lower() == ".png":
                        new.save(dst)
                    else:
                        new.save(dst, quality=quality)
                    shrunk += 1
                else:
                    shutil.copy2(img, dst)
                    copied += 1
            lbl = lbl_dir / (img.stem + ".txt")
            if lbl.exists():
                shutil.copy2(lbl, out / split / "labels" / lbl.name)
                labels += 1
            if i % 1000 == 0:
                print(f"  {split}: {i}/{len(files)}", flush=True)
        print(f"{split}: {len(files)} images done")

    for extra in ("manifest.csv",):
        if (src / extra).exists():
            shutil.copy2(src / extra, out / extra)

    print(f"\nShrunk {shrunk} images, copied {copied} as they were, {labels} label files.")
    print("Zipping...")
    shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
    mb = zip_path.stat().st_size / 1e6
    print(f"Created {zip_path}  ({mb:.0f} MB)")
    print("Upload that zip to Google Drive (folder: helmet).")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True, help="Merged dataset folder, e.g. ..\\dataset_merged")
    p.add_argument("--out", required=True, help="New smaller folder, e.g. ..\\dataset_merged_small")
    p.add_argument("--max-side", type=int, default=1280, help="Longest side in pixels (default 1280)")
    p.add_argument("--quality", type=int, default=92, help="JPEG quality for shrunk images (default 92)")
    a = p.parse_args()
    main(Path(a.src), Path(a.out), a.max_side, a.quality)