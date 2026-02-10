import os
import glob
import subprocess
import sys
import zipfile
import shutil

DESKTOP = os.path.expanduser("~/Desktop")

SPEAKER_SCRIPT = os.path.join(DESKTOP, "ellie_participant_split.py")
AUS_SCRIPT = os.path.join(DESKTOP, "aus_split.py")

# Remove the zipped folders after successful extraction
DELETE_ZIPS_AFTER_EXTRACT = True

def run(cmd, cwd):
    print("\n>>>", " ".join(cmd))
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        print("!! Failed in:", cwd)
        sys.exit(r.returncode)

def _flatten_single_nested_dir(target_dir: str):
    """
    Some zips extract into a single nested folder (e.g., target_dir/300_P/...).
    If target_dir contains exactly one directory and no files, move its contents up.
    """
    entries = [e for e in os.listdir(target_dir) if e not in (".DS_Store",)]
    paths = [os.path.join(target_dir, e) for e in entries]
    dirs = [p for p in paths if os.path.isdir(p)]
    files = [p for p in paths if os.path.isfile(p)]

    if len(dirs) == 1 and len(files) == 0:
        nested = dirs[0]
        for item in os.listdir(nested):
            src = os.path.join(nested, item)
            dst = os.path.join(target_dir, item)
            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            shutil.move(src, dst)
        shutil.rmtree(nested)

def unzip_archives_on_desktop():
    zips = sorted(glob.glob(os.path.join(DESKTOP, "*_P.zip")))
    if not zips:
        return

    print("Found zip archives:", len(zips))
    for zip_path in zips:
        base = os.path.splitext(os.path.basename(zip_path))[0]  # e.g., "300_P"
        out_dir = os.path.join(DESKTOP, base)

        # If folder exists and seems non-empty, skip extraction but still delete zip (optional).
        # Here: only delete zip if we actually extract it successfully.
        if os.path.isdir(out_dir) and os.listdir(out_dir):
            print(f" - {base} already extracted, skipping unzip (keeping zip).")
            continue

        os.makedirs(out_dir, exist_ok=True)
        print(f" - Unzipping {os.path.basename(zip_path)} -> {out_dir}")

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)

        _flatten_single_nested_dir(out_dir)

        if DELETE_ZIPS_AFTER_EXTRACT:
            os.remove(zip_path)
            print(f" Deleted zip: {zip_path}")


def cleanup_folder_keep_only(folder: str, keep_paths):
    """
    Delete EVERYTHING inside folder except the files listed in keep_paths.
    Also removes empty directories afterward.
    """
    keep_abs = set(os.path.abspath(p) for p in keep_paths)

    deleted = 0
    for root, dirs, files in os.walk(folder, topdown=False):
        for f in files:
            p = os.path.abspath(os.path.join(root, f))
            if p not in keep_abs:
                try:
                    os.remove(p)
                    deleted += 1
                except Exception as e:
                    print(" !! Could not delete file:", p, "->", e)

        for d in dirs:
            dp = os.path.join(root, d)
            try:
                if not os.listdir(dp):
                    os.rmdir(dp)
            except Exception:
                pass

    print(f"Cleanup done in {os.path.basename(folder)}. Deleted {deleted} file(s).")


def main():
    # 1) unzip first (and delete zips after extracting)
    unzip_archives_on_desktop()

    # 2) Process extracted folders like 300_P, 301_P, ...
    folders = sorted(
        [
            os.path.join(DESKTOP, d)
            for d in os.listdir(DESKTOP)
            if os.path.isdir(os.path.join(DESKTOP, d)) and d.endswith("_P")
        ]
    )

    if not folders:
        print("No *_P folders found on Desktop.")
        return

    for folder in folders:
        folder_name = os.path.basename(folder)
        print("\n==============================")
        print("Processing:", folder_name)
        print("==============================")

        prefix = folder_name.split("_", 1)[0]  # "300_P" -> "300"

        transcripts = glob.glob(os.path.join(folder, "**", "*_TRANSCRIPT.csv"), recursive=True)
        aus_files = glob.glob(os.path.join(folder, "**", "*_CLNF_AUs.txt"), recursive=True)

        if not transcripts:
            print(" - No *_TRANSCRIPT.csv found, skipping.")
            continue
        if not aus_files:
            print(" - No *_CLNF_AUs.txt found, skipping.")
            continue

        transcript = transcripts[0]
        aus_file = aus_files[0]

        segments_out = os.path.join(folder, f"{prefix}_speaker_segments.csv")
        labeled_out = os.path.join(folder, f"{prefix}_CLNF_AUs_labeled.csv")

        run(["python", SPEAKER_SCRIPT, transcript, "-o", segments_out], cwd=folder)
        run(["python", AUS_SCRIPT, segments_out, aus_file, "-o", labeled_out], cwd=folder)

        print("Done:", folder_name)
        print("Kept core files + outputs, now cleaning up...")

        cleanup_folder_keep_only(
            folder,
            keep_paths=[transcript, aus_file, segments_out, labeled_out],
        )

        print("Final kept files:")
        print(" -", transcript)
        print(" -", aus_file)
        print(" -", segments_out)
        print(" -", labeled_out)


if __name__ == "__main__":
    main()