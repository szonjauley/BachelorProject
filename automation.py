import os
import glob
import subprocess
import sys

DESKTOP = os.path.expanduser("~/Desktop")

SPEAKER_SCRIPT = os.path.join(DESKTOP, "ellie_participant_split.py")
AUS_SCRIPT = os.path.join(DESKTOP, "aus_split.py")

def run(cmd, cwd):
    print("\n>>>", " ".join(cmd))
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        print("!! Failed in:", cwd)
        sys.exit(r.returncode)

def main():
    # Folders like 300_P, 301_P, ...
    folders = sorted(
        [os.path.join(DESKTOP, d) for d in os.listdir(DESKTOP)
         if os.path.isdir(os.path.join(DESKTOP, d)) and d.endswith("_P")]
    )

    if not folders:
        print("No *_P folders found on Desktop.")
        return

    for folder in folders:
        folder_name = os.path.basename(folder)
        print("\n==============================")
        print("Processing:", folder_name)
        print("==============================")

        # Extract prefix number from folder name like "300_P" -> "300"
        prefix = folder_name.split("_", 1)[0]

        # Find transcript + AU files in that folder
        transcripts = glob.glob(os.path.join(folder, "*_TRANSCRIPT.csv"))
        aus_files = glob.glob(os.path.join(folder, "*_CLNF_AUs.txt"))

        if not transcripts:
            print(" - No *_TRANSCRIPT.csv found, skipping.")
            continue
        if not aus_files:
            print(" - No *_CLNF_AUs.txt found, skipping.")
            continue

        transcript = transcripts[0]
        aus_file = aus_files[0]

        # Output paths inside the same folder (with numeric prefix)
        segments_out = os.path.join(folder, f"{prefix}_speaker_segments.csv")
        labeled_out = os.path.join(folder, f"{prefix}_CLNF_AUs_labeled.csv")

        # Make speaker segments
        run(
            ["python", SPEAKER_SCRIPT, transcript, "-o", segments_out],
            cwd=folder
        )

        # Label AUs by speaker
        run(
            ["python", AUS_SCRIPT, segments_out, aus_file, "-o", labeled_out],
            cwd=folder
        )

        print("Done:", folder_name)
        print("Speaker segments:", segments_out)
        print("AUs labeled:", labeled_out)

if __name__ == "__main__":
    main()