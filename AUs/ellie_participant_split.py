import argparse
import os
import pandas as pd
import csv




def main():
    # Create the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv") # input file (e.g., XXX_TRANSCRIPT.csv)
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()

    input_path = args.input_csv

    # In case we don't provide an output file name it will be XXX_TRANSCRIPT_speakers_segments.csv
    if args.output is None:
        base, _ = os.path.splitext(os.path.basename(input_path))
        out_path = f"{base}_speakers_segments.csv"
    else:
        out_path = args.output

    with open(input_path, "r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=["\t", ";", ","])
            detected_sep = dialect.delimiter
        except Exception:
            detected_sep = "\t" if "\t" in sample else (";" if ";" in sample else ",")

    input = pd.read_csv(input_path, sep=detected_sep)

    # input = pd.read_csv(input_path, sep=";")

    input.columns = input.columns.str.strip() # Remove spaces in columns names

    # Check if all required columns are present; otherwise, give an error
    required_columns_input = {"speaker", "start_time", "stop_time", "value"}
    if not required_columns_input.issubset(input.columns):
        raise ValueError(f"Missing required columns {required_columns_input}. Found: {list(input.columns)}")

    # Convert start_time and stop_time to numeric, if conversion fails we give NaN
    input["start_time"] = pd.to_numeric(input["start_time"], errors="coerce")
    input["stop_time"] = pd.to_numeric(input["stop_time"], errors="coerce")

    # Drop rows that are missing the specified columns as it is not possible to create segments without them
    input = input.dropna(subset=["speaker", "start_time", "stop_time"]).copy()

    # Sort transcript rows by start_time so consecutive turns are in the right order
    input = input.sort_values("start_time").reset_index(drop=True)

    # Create a group id that increments every time the speaker changes from the previous row
    # (e.g., Ellie Ellie Ellie Participant Participant Ellie correspond to groups 1 1 1 2 2 3)
    group_id = (input["speaker"] != input["speaker"].shift()).cumsum()

    segments = (
        input.groupby(group_id)
          .agg(
              speaker=("speaker", "first"), # speaker name for this "run"
              start_time=("start_time", "first"), # first start_time in the "run"
              stop_time=("stop_time", "last"), # last stop_time in the "run"
              text=("value", lambda x: " ".join(str(s).strip() for s in x if pd.notna(s))) # concatenation of all transcript "value" cells in that run
          )
          .reset_index(drop=True)
    )

    segments.to_csv(out_path, index=False, sep=";")

    print(f"Done! Output file name is {out_path}")
    print(f"It contains a total number of segments of {len(segments)}")

if __name__ == "__main__":
    main()