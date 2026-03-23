from pathlib import Path
import re
import pandas as pd
import numpy as np


# Directory where the current script is located
SCRIPT_DIR = Path(__file__).parent.resolve()

ROOT_DIR = SCRIPT_DIR / "participant_folders"
FULL_TEST_PATH = SCRIPT_DIR / "depression.csv"
OUTPUT_PATH = SCRIPT_DIR / "au_long.csv"

# Minimum confidence threshold for keeping the frames
CONF_THRESH = 0.7
# If True, keep only rows where success == 1
REQUIRE_SUCCESS = True
# Speaker codes
SPEAKING_CODE = 1
LISTENING_CODE = 0

# Load labels
labels = pd.read_csv(FULL_TEST_PATH, sep=None, engine="python")
# Remove accidental spaces from column names
labels.columns = labels.columns.str.strip()


def pick_col(df, candidates):
    """Return the first matching column name from a list of candidate names."""
    cols = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(" ", "").replace("_", "")
        if key in cols:
            return cols[key]
    raise KeyError(f"None of {candidates} found in columns: {list(df.columns)}")


# Identify the participant ID column and the binary depression label column
pid_col = pick_col(labels, ["Participant_ID", "ParticipantID"])
bin_col = pick_col(labels, ["PHQ_Binary", "PHQBinary", "PHQ8_Binary"])

labels[pid_col] = labels[pid_col].astype(int)
# Create a dictionary where the key is the participant ID and the value is its depression status
id2dep = dict(zip(labels[pid_col], labels[bin_col]))

# Statistics to compute for each AU intensity column (*_r)
R_STATS = {
    "mean": np.mean,
    "std":  np.std,
}

def apply_stats(x, stats):
    """Apply a dictionary of statistic functions to a vector x"""
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().values
    if len(x) == 0:
        return {k: np.nan for k in stats}
    out = {}
    for k, f in stats.items():
        try:
            out[k] = f(x)
        except Exception:
            out[k] = np.nan
    return out


def infer_masks(df: pd.DataFrame):
    """
    Create boolean masks for:
    - all frames;
    - speaking frames;
    - listening frames.

    Returns a dictionary of masks
    """
    if "speaker" not in df.columns:
        raise KeyError("Your AU file has no 'speaker' column.")

    sp = df["speaker"]

    sp_num = pd.to_numeric(sp, errors="coerce")
    numeric_fraction = sp_num.notna().mean()

    # If almost all values are numeric, use numeric speaker codes
    if numeric_fraction > 0.95:
        speaking = sp_num == SPEAKING_CODE
        listening = sp_num == LISTENING_CODE
    else:
        # Otherwise treat speaker labels as strings
        s = sp.astype(str).str.strip().str.lower()
        listening = s.eq("listening") | s.str.contains("listen")
        speaking  = s.eq("speaking") | s.str.contains("speak")

        if listening.sum() > 0 and speaking.sum() == 0:
            speaking = ~listening
        elif speaking.sum() > 0 and listening.sum() == 0:
            listening = ~speaking

    if listening.sum() == 0 and speaking.sum() == 0:
        uniques = pd.Series(sp.dropna().unique()).astype(str).tolist()
        raise RuntimeError(
            "Could not create ANY masks.\n"
            f"speaker dtype: {sp.dtype}\n"
            f"unique speaker values: {uniques}\n"
        )

    allmask = pd.Series(True, index=df.index)
    masks = {"all": allmask}
    if speaking.sum() > 0:
        masks["speaking"] = speaking
    if listening.sum() > 0:
        masks["listening"] = listening
    return masks


# Discover AU files recursively
root = Path(ROOT_DIR)
au_files = sorted(root.rglob("*_CLNF_AUs_labeled.csv"))
print("Found AU files:", len(au_files))

if len(au_files) == 0:
    raise RuntimeError("No *_CLNF_AUs_labeled.csv files found. ROOT_DIR is wrong.")

rows = []
printed_speaker_values = False
printed_segment_counts = False

for au_file in au_files:
    # Extract participant ID from filename, e.g. "123_CLNF_AUs_labeled.csv" -> 123
    m = re.match(r"^(\d+)_CLNF_AUs_labeled\.csv$", au_file.name)
    if not m:
        continue
    pid = int(m.group(1))

    df = pd.read_csv(au_file)
    df.columns = df.columns.str.strip()

    if not printed_speaker_values:
        print("Speaker unique values (first file):",
              pd.Series(df["speaker"].dropna().unique()).astype(str).tolist()[:30])
        print("Frames before filters:", len(df))
        printed_speaker_values = True

    # Keep only SUCCESS frames, if requested
    if REQUIRE_SUCCESS and "success" in df.columns:
        df = df[df["success"] == 1]
    # Keep only confident frames, if requested
    if CONF_THRESH is not None and "confidence" in df.columns:
        df = df[df["confidence"] >= CONF_THRESH]

    if printed_speaker_values:
        print("Frames after filters (first file):", len(df))
        printed_speaker_values = False

    if len(df) == 0:
        continue

    depressed = id2dep.get(pid, np.nan)

    au_r_cols = [c for c in df.columns if c.startswith("AU") and c.endswith("_r")]

    masks = infer_masks(df)

    if "speaking" not in masks or "listening" not in masks:
        missing = "listening" if "speaking" in masks else "speaking"
        print(f"PID {pid} — missing '{missing}' segment, only has: {[k for k in masks if k != 'all']}")

    if not printed_segment_counts:
        print("Example segment counts:", {k: int(v.sum()) for k, v in masks.items()})
        printed_segment_counts = True

    for segment_type, mask in masks.items():
        seg = df[mask]
        if len(seg) == 0:
            continue

        for col in au_r_cols:
            stats = apply_stats(seg[col], R_STATS)
            for stat, val in stats.items():
                rows.append({
                    "person_id": pid,
                    "depressed": depressed,
                    "segment_type": segment_type,
                    "AU": col,
                    "stat": stat,
                    "value": float(val) if pd.notna(val) else np.nan
                })

# Create final long-format DataFrame
out = pd.DataFrame(rows, columns=["person_id", "depressed", "segment_type", "AU", "stat", "value"])
print("Produced rows:", len(out))

if out.empty:
    raise RuntimeError(
        "Produced 0 rows. Debug by setting CONF_THRESH=None and REQUIRE_SUCCESS=False.\n"
        "Also verify that filters are not removing all frames."
    )

out["depressed"] = pd.to_numeric(out["depressed"], errors="coerce")
out.to_csv(OUTPUT_PATH, index=False)
print("Saved:", OUTPUT_PATH)