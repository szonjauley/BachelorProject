import pandas as pd
from pathlib import Path

# Directory where the current script is located
SCRIPT_DIR = Path(__file__).parent.resolve() # BachelorProject/gaze
DATA_DIR = SCRIPT_DIR.parent / "data" # BachelorProject/data

def load_data(file: Path) -> pd.DataFrame:
    """
    Loading gaze delta values file
    """
    return pd.read_csv(file)

def aggregate_file(df: pd.DataFrame, segment: str) -> pd.DataFrame:
    """
    Aggregate values for a single file and format output
    """
    # get depression label per person
    depression_map = df.groupby("person_id")["depressed"].first()

    # compute stats
    stats = (
        df.groupby("person_id")["delta_deg"]
        .agg(["mean", "std"])
        .reset_index()
    )

    # convert to long format
    stats = stats.melt(
        id_vars="person_id",
        value_vars=["mean", "std"],
        var_name="stat",
        value_name="value"
    )

    # attach depression label
    stats["depressed"] = stats["person_id"].map(depression_map)

    # add segment label
    stats["segment_type"] = segment

    # reorder columns
    stats = stats[["person_id", "stat", "depressed", "segment_type", "value"]]

    return stats

def combine_files(files:list, segments:list) -> pd.DataFrame:
    """
    Combine aggregated files into a single output
    """
    dfs = [
        aggregate_file(load_data(file), segment)
        for file, segment in zip(files, segments)
    ]

    combined = pd.concat(dfs, ignore_index=True)

    # order by person_ID
    combined = combined.sort_values(["person_id", "segment_type", "stat"])

    return combined

def main():

    files = [
        DATA_DIR / "combined_gaze_deltas.csv",
        DATA_DIR / "listening_gaze_deltas.csv",
        DATA_DIR / "speaking_gaze_deltas.csv"
    ]

    segments = ["all", "listening", "speaking"]

    combined = combine_files(files, segments)

    output_path = DATA_DIR / "gaze_aggregation.csv"
    combined.to_csv(output_path, index=False)

    return

if __name__ == "__main__":
    main()