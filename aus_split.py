import argparse
import os
import numpy as np
import pandas as pd




def label_timestamps_with_segments(au_df: pd.DataFrame, segments_df: pd.DataFrame) -> pd.DataFrame:
    # Using copies to not modify the orginal data
    au_df = au_df.copy()
    segments_df = segments_df.copy()
    # Remove spaces in columns names
    au_df.columns = au_df.columns.str.strip()
    segments_df.columns = segments_df.columns.str.strip()

    # Check if all required columns are present; otherwise, give an error
    needed_columns_segments_df = {"speaker", "start_time", "stop_time"}
    if not needed_columns_segments_df.issubset(segments_df.columns):
        raise ValueError(
            f"Segments file must contain columns: {needed_columns_segments_df}. "
            f"Found: {list(segments_df.columns)}"
        )

    au_df["timestamp_raw"] = au_df["timestamp"].astype(str)
    au_df["timestamp_num"] = pd.to_numeric(au_df["timestamp_raw"].str.strip(), errors="coerce")

    # Convert segment boundaries to numeric
    segments_df["start_time"] = pd.to_numeric(segments_df["start_time"], errors="coerce")
    segments_df["stop_time"] = pd.to_numeric(segments_df["stop_time"], errors="coerce")

    # Drop all AU rows with no valid timestamp
    au_df = au_df.dropna(subset=["timestamp_num"]).copy()
    segments_df = segments_df.dropna(subset=["speaker", "start_time", "stop_time"]).copy()

    segments_df = segments_df.sort_values("start_time").reset_index(drop=True)

    starts = segments_df["start_time"].to_numpy() # Start time of each segment
    stops = segments_df["stop_time"].to_numpy() # Stop time of each segment
    speakers = segments_df["speaker"].astype(str).to_numpy() # Speaker label for each segment

    # For each timestamp t, find the last segment whose start_time <= t
    timestamp = au_df["timestamp_num"].to_numpy()
    idx = np.searchsorted(starts, timestamp, side="right") - 1

    # idx may be -1 for timestamps that come before the first segment, clip to avoid indexing errors when checking stops
    idx_clip = np.clip(idx, 0, len(stops) - 1)

    valid_timestamp = (idx >= 0) & (timestamp <= stops[idx_clip])

    out = au_df.loc[valid_timestamp].copy()

    # New first column "speaker" using the matched segment speaker labels
    out.insert(0, "speaker", speakers[idx[valid_timestamp]])
    # Convert DAIC-WOZ names to Listening (Ellie) and Speaking (Participant)
    out["speaker"] = out["speaker"].replace({
        "Ellie": "Listening",
        "Participant": "Speaking",
    })

    # Restore the original timestamp string to avoid formatting changes
    out["timestamp"] = out["timestamp_raw"]
    out = out.drop(columns=["timestamp_raw", "timestamp_num"])

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("segments_csv")
    parser.add_argument("aus_txt")
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()

    if args.output is None:
        base, _ = os.path.splitext(os.path.basename(args.aus_txt))
        out_path = f"{base}_labeled.csv"
    else:
        out_path = args.output

    segments_df = pd.read_csv(args.segments_csv, sep=";")
    segments_df.columns = segments_df.columns.str.strip()

    au_df = pd.read_csv(args.aus_txt, sep=",", engine="python", dtype=str)
    au_df.columns = au_df.columns.str.strip()

    labeled = label_timestamps_with_segments(au_df, segments_df)

    labeled.to_csv(out_path, index=False)

    print(f"Done! Output file name is {out_path}")
    print("Kept rows:", len(labeled), "of", len(au_df))

if __name__ == "__main__":
    main()