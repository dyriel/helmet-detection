"""
inspect_myanmar_clip.py

Prints the structure of an extracted Myanmar HELMET folder (e.g. test_v2) so
the converter can be written against the real layout. Read-only.

Shows: the clip folders found, a sample clip's contents, the first lines of
one annotation file, and the size of one image.

Usage (run from the scripts folder):
    python inspect_myanmar_clip.py --root ..\\raw_myanmar\\test_v2
"""

import argparse
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png"}


def main(root: Path):
    if not root.exists():
        raise SystemExit(f"Folder not found: {root}")

    clips = sorted(p for p in root.iterdir() if p.is_dir())
    print(f"Root: {root}")
    print(f"Clip folders found: {len(clips)}")
    print("First 10:", [c.name for c in clips[:10]], "\n")
    if not clips:
        print("No subfolders. Is there another folder level inside? Contents:")
        for p in list(root.iterdir())[:20]:
            print("  ", p.name)
        return

    clip = clips[0]
    print("=" * 60)
    print(f"Sample clip: {clip.name}")
    print("=" * 60)
    for p in sorted(clip.iterdir())[:20]:
        kind = "folder" if p.is_dir() else "file"
        print(f"  [{kind}] {p.name}")

    img_files = sorted(f for f in clip.rglob("*") if f.suffix.lower() in IMG_EXT)
    ann_files = sorted(
        f for f in clip.rglob("*")
        if f.is_file() and f.suffix.lower() not in IMG_EXT
    )
    print(f"\nImage files in this clip: {len(img_files)}")
    for f in img_files[:5]:
        print("  ", f.relative_to(clip))
    print(f"Non-image files in this clip: {len(ann_files)}")
    for f in ann_files[:5]:
        print("  ", f.relative_to(clip))

    if ann_files:
        f = ann_files[0]
        print("\n" + "=" * 60)
        print(f"First 8 lines of {f.name}:")
        print("=" * 60)
        try:
            with open(f, encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh):
                    if i >= 8:
                        break
                    print(line.rstrip())
        except Exception as e:
            print(f"(could not read: {e})")

    if img_files:
        try:
            from PIL import Image
            with Image.open(img_files[0]) as im:
                print(f"\nFirst image size: {im.size[0]} x {im.size[1]}  ({img_files[0].name})")
        except ImportError:
            print("\n(pip install pillow --break-system-packages to also print image size)")
        except Exception as e:
            print(f"\n(could not read image size for {img_files[0].name}: {e})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, help="Extracted folder, e.g. raw_myanmar\\test_v2")
    main(Path(p.parse_args().root))