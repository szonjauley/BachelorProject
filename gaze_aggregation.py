import pandas as pd
from pathlib import Path


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
    depression_map = df.groupby("person_ID")["depression"].first()

    # compute stats
    stats = (
        df.groupby("person_ID")["delta_deg"]
        .agg(["mean", "std"])
        .reset_index()
    )

    # convert to long format
    stats = stats.melt(
        id_vars="person_ID",
        value_vars=["mean", "std"],
        var_name="stat",
        value_name="delta_deg"
    )

    # attach depression label
    stats["depression"] = stats["person_ID"].map(depression_map)

    # add segment label
    stats["segment"] = segment

    # reorder columns
    stats = stats[["person_ID", "stat", "depression", "segment", "delta_deg"]]

    return stats

def combine_files(files, segments):
    """
    Combine aggregated files into a single output
    """
    dfs = [
        aggregate_file(load_data(file), segment)
        for file, segment in zip(files, segments)
    ]

    combined = pd.concat(dfs, ignore_index=True)

    # order by person_ID
    combined = combined.sort_values(["person_ID", "segment", "stat"])

    return combined