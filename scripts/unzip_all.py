"""
unzip_all.py

Unzips every .zip file found under a folder (including zips inside zips),
then deletes each zip -- but only after checking the extraction worked.

Safety checks per zip:
  1. The zip passes an integrity test (no corrupt data).
  2. Every file inside it exists on disk afterward, at the expected size.
  3. No file inside the zip is allowed to land outside the target folder.
If any check fails, that zip is kept and reported, and nothing is deleted.

Where files go: if the zip already contains one top-level folder, it is
extracted next to the zip. Otherwise a folder named after the zip is created.
So Bago_highway_11.zip never ends up as Bago_highway_11\\Bago_highway_11\\.

Usage (run from the scripts folder):
    python unzip_all.py --root ..\\raw_myanmar\\test_v2 --dry-run   (just shows what it would do)
    python unzip_all.py --root ..\\raw_myanmar\\test_v2             (does it)
    python unzip_all.py --root ..\\raw_myanmar\\test_v2 --keep-zips (extract, don't delete)
"""

import argparse
import shutil
import zipfile
from pathlib import Path


def top_level_names(names):
    tops = set()
    for n in names:
        parts = [p for p in n.replace("\\", "/").split("/") if p]
        if parts:
            tops.add(parts[0])
    return tops


def process_zip(zip_path: Path, keep: bool, dry: bool) -> bool:
    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile:
        print(f"  FAILED (not a valid zip): {zip_path}")
        return False

    with zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        names = [i.filename for i in zf.infolist()]
        tops = top_level_names(names)

        # one top-level folder inside -> extract beside the zip; else make a folder
        single_top = len(tops) == 1 and any(
            "/" in n.replace("\\", "/") for n in names
        )
        dest = zip_path.parent if single_top else zip_path.parent / zip_path.stem
        dest_resolved = dest.resolve()

        print(f"{zip_path.name}: {len(infos)} files -> {dest}")
        if dry:
            return True

        bad = zf.testzip()
        if bad is not None:
            print(f"  FAILED (corrupt entry {bad}), zip kept")
            return False

        for info in infos:
            target = (dest / info.filename).resolve()
            if dest_resolved not in target.parents and target != dest_resolved:
                print(f"  FAILED (unsafe path in zip: {info.filename}), zip kept")
                return False

        dest.mkdir(parents=True, exist_ok=True)
        zf.extractall(dest)

        for info in infos:
            out = dest / info.filename
            if not out.exists() or out.stat().st_size != info.file_size:
                print(f"  FAILED (missing or wrong size: {info.filename}), zip kept")
                return False

    if not keep:
        zip_path.unlink()
        print("  extracted OK, zip deleted")
    else:
        print("  extracted OK, zip kept")
    return True


def main(root: Path, keep: bool, dry: bool):
    if not root.exists():
        raise SystemExit(f"Folder not found: {root}")

    total_ok = 0
    failed = []
    for pass_num in range(1, 6):  # handles zips found inside extracted zips
        zips = sorted(root.rglob("*.zip"))
        if dry and pass_num > 1:
            break
        # skip ones that already failed so we don't loop on them forever
        zips = [z for z in zips if z not in failed]
        if not zips:
            break
        print(f"--- pass {pass_num}: {len(zips)} zip file(s) ---")
        for z in zips:
            if process_zip(z, keep, dry):
                total_ok += 1
            else:
                failed.append(z)
        if keep or dry:
            break  # zips are still there, a second pass would repeat them

    print(f"\nDone. {total_ok} zip(s) processed OK, {len(failed)} failed.")
    for f in failed:
        print(f"  kept (failed): {f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, help="Folder to search for .zip files")
    p.add_argument("--keep-zips", action="store_true", help="Don't delete zips after extracting")
    p.add_argument("--dry-run", action="store_true", help="Only show what would happen")
    a = p.parse_args()
    main(Path(a.root), a.keep_zips, a.dry_run)