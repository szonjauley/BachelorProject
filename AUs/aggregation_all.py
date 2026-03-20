from pathlib import Path
import re
import pandas as pd
import numpy as np

ROOT_DIR = "/Users/pietrorebecchi/Desktop/participant_folders"
FULL_TEST_PATH = "/Users/pietrorebecchi/Desktop/ids.csv"
OUTPUT_PATH = "/Users/pietrorebecchi/Desktop/au_long.csv"

CONF_THRESH = 0.7
REQUIRE_SUCCESS = True

SPEAKING_CODE = 1
LISTENING_CODE = 0


# Load labels
labels = pd.read_csv(FULL_TEST_PATH, sep=None, engine="python")
labels.columns = labels.columns.str.strip()

def pick_col(df, candidates):
    cols = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(" ", "").replace("_", "")
        if key in cols:
            return cols[key]
    raise KeyError(f"None of {candidates} found in columns: {list(df.columns)}")

pid_col = pick_col(labels, ["Participant_ID", "ParticipantID"])
bin_col = pick_col(labels, ["PHQ_Binary", "PHQBinary"])

labels[pid_col] = labels[pid_col].astype(int)
id2dep = dict(zip(labels[pid_col], labels[bin_col]))


# Stats helpers
def q(p):
    return lambda x: np.quantile(x, p)

def mad(x):
    med = np.median(x)
    return np.median(np.abs(x - med))

# intensity AUs (*_r)
R_STATS = {
    "count":  lambda x: len(x),
    "min":    np.min,
    "q05":    q(0.05),
    "q25":    q(0.25),
    "median": np.median,
    "q75":    q(0.75),
    "q95":    q(0.95),
    "max":    np.max,
    "iqr":    lambda x: np.quantile(x, 0.75) - np.quantile(x, 0.25),
    "mean":   np.mean,
    "std":    np.std,
}

# binary AUs (*_c)
C_STATS = {
    "count": lambda x: len(x),
    "mean":  np.mean,
    "std":   np.std,
}

def safe_apply_stats(x, stats):
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


# Robust Speaking/Listening masks (works for strings OR numeric)
def infer_masks(df: pd.DataFrame):
    if "speaker" not in df.columns:
        raise KeyError("Your AU file has no 'speaker' column.")

    sp = df["speaker"]

    # Try numeric conversion; only treat as numeric if almost all values convert.
    sp_num = pd.to_numeric(sp, errors="coerce")
    numeric_fraction = sp_num.notna().mean()

    if numeric_fraction > 0.95:
        # numeric codes
        speaking = sp_num == SPEAKING_CODE
        listening = sp_num == LISTENING_CODE
    else:
        # string/categorical labels like "Listening"/"Speaking"
        s = sp.astype(str).str.strip().str.lower()
        listening = s.eq("listening") | s.str.contains("listen")
        speaking  = s.eq("speaking")  | s.str.contains("speak")

        # If only one side matched, infer the other as the complement
        if listening.sum() > 0 and speaking.sum() == 0:
            speaking = ~listening
        elif speaking.sum() > 0 and listening.sum() == 0:
            listening = ~speaking

    # sanity: we expect both to exist in your dataset
    if listening.sum() == 0 or speaking.sum() == 0:
        uniques = pd.Series(sp.dropna().unique()).astype(str).tolist()
        raise RuntimeError(
            "Could not create BOTH listening and speaking masks.\n"
            f"speaker dtype: {sp.dtype}\n"
            f"unique speaker values: {uniques}\n"
            f"numeric_fraction={numeric_fraction:.3f} (numeric branch only if > 0.95)\n"
            "If speaker is numeric, set SPEAKING_CODE/LISTENING_CODE correctly."
        )

    allmask = pd.Series(True, index=df.index)
    return {"all": allmask, "speaking": speaking, "listening": listening}


# Discover AU files recursively 
root = Path(ROOT_DIR)
au_files = au_files = sorted(root.rglob("*_CLNF_AUs_labeled.csv"))
print("Found AU files:", len(au_files))
print("Example:", au_files[:3])

if len(au_files) == 0:
    raise RuntimeError("No *_CLNF_AUs_labeled.csv files found. ROOT_DIR is wrong.")

rows = []
printed_speaker_values = False
printed_segment_counts = False

for au_file in au_files:
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

    # filters
    if REQUIRE_SUCCESS and "success" in df.columns:
        df = df[df["success"] == 1]
    if CONF_THRESH is not None and "confidence" in df.columns:
        df = df[df["confidence"] >= CONF_THRESH]

    if printed_speaker_values:
        print("Frames after filters (first file):", len(df))
        printed_speaker_values = False

    if len(df) == 0:
        continue

    depressed = id2dep.get(pid, np.nan)

    au_r_cols = [c for c in df.columns if c.startswith("AU") and c.endswith("_r")]
    au_c_cols = [c for c in df.columns if c.startswith("AU") and c.endswith("_c")]

    masks = infer_masks(df)

    # print segment counts once to confirm listening/speaking are found
    if not printed_segment_counts:
        print("Example segment counts:", {k: int(v.sum()) for k, v in masks.items()})
        printed_segment_counts = True

    for segment_type, mask in masks.items():
        seg = df[mask]
        if len(seg) == 0:
            continue

        # intensity AUs
        for col in au_r_cols:
            stats = safe_apply_stats(seg[col], R_STATS)
            for stat, val in stats.items():
                rows.append({
                    "person_id": pid,
                    "depressed": depressed,
                    "segment_type": segment_type,
                    "AU": col,
                    "stat": stat,
                    "value": float(val) if pd.notna(val) else np.nan
                })

        # binary AUs
        for col in au_c_cols:
            x = pd.to_numeric(seg[col], errors="coerce").dropna()
            if len(x) == 0:
                continue
            x = (x > 0).astype(int)

            stats = safe_apply_stats(x, C_STATS)
            for stat, val in stats.items():
                rows.append({
                    "person_id": pid,
                    "depressed": depressed,
                    "segment_type": segment_type,
                    "AU": col,
                    "stat": stat,
                    "value": float(val) if pd.notna(val) else np.nan
                })

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