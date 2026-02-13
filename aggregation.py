import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def load_all_data(base_folder:Path, speaker:str="all") -> pd.DataFrame:
    """
    Loads all AU csv files into one dataframe with a person_ID column.
    Takes folder path and speaker mode as input and returns dataframe with additional ID column
    """
    base_path = Path(base_folder)
    dfs = []

    for person_folder in sorted(base_path.glob("*_P")):
        person_id = person_folder.name.split("_")[0]
        file_path = person_folder / f"{person_id}_CLNF_AUs_labeled.csv"

        if not file_path.exists():
            print(f"Missing file for {person_id}")
            continue

        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()

        df["person_ID"] = person_id
        dfs.append(df)

    if not dfs:
        raise ValueError("No AU files found.")

    data = pd.concat(dfs, ignore_index=True)

    # Speaker mode filter 
    if speaker == "listening":
        data = data[data["speaker"] == "Listening"]
    elif speaker == "speaking":
        data = data[data["speaker"] == "Speaking"]

    return data


def compute_person_stats(data:pd.DataFrame, confidence:float=0.9) ->pd.DataFrame:
    """
    Takes the combined data, filters it for the specified confidence and aggregates over the accepted frames
    Returns the stats for each person
    """

    data = data[
        (data["confidence"] > confidence) &
        (data["success"] == 1)
    ]

    au_cols = [c for c in data.columns if c.startswith("AU") and c.endswith("_r")]

    stats = (
        data
        .groupby("person_ID")[au_cols]
        .agg(
            mean="mean",
            std="std",
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
            min="min",
            max="max"
        )
    )

    # Flatten MultiIndex columns
    stats.columns = [f"{au}_{stat}" for au, stat in stats.columns]

    return stats


def compute_group_stats(person_stats:pd.DataFrame) ->pd.DataFrame:
    """
    Takes person level stats and computes population-level statistics from person-level means
    Prevents participants with more frames from biasing results
    """

    # Keep only the mean columns
    mean_cols = [c for c in person_stats.columns if c.endswith("_mean")]

    group_stats = (
        person_stats[mean_cols]
        .agg(
            group_mean="mean",
            group_std="std",
            group_median="median",
            group_q25=lambda x: x.quantile(0.25),
            group_q75=lambda x: x.quantile(0.75),
            group_min="min",
            group_max="max"
        )
        .T
    )

    group_stats.columns = [
        "group_mean",
        "group_std",
        "group_median",
        "group_q25",
        "group_q75",
        "group_min",
        "group_max"
    ]

    return group_stats

def main(base_folder, speaker="all", confidence=0.9):

    print("Loading data...")
    combined = load_all_data(base_folder, speaker)

    print("Computing person-level statistics...")
    person_stats = compute_person_stats(combined, confidence)

    print("Computing group-level statistics...")
    group_stats = compute_group_stats(person_stats)

    # Save
    combined_file = "combined_AUs_labeled.csv"
    person_file = "AU_stats_person_level.csv"
    group_file = "AU_stats_group_level.csv"

    combined.to_csv(combined_file)
    person_stats.to_csv(person_file)
    group_stats.to_csv(group_file)

    print(f"\nSaved:")
    print(combined_file)
    print(person_file)
    print(group_file)

    return combined, person_stats, group_stats

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Aggregate AU scores at person and group level."
    )

    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Path to base folder containing person directories"
    )

    parser.add_argument(
        "--speaker",
        type=str,
        default="all",
        choices=["all", "listening", "speaking"],
        help="Speaker filter"
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.9,
        help="Minimum confidence threshold"
    )

    args = parser.parse_args()

    main(
        base_folder=args.folder,
        speaker=args.speaker,
        confidence=args.confidence
    )

