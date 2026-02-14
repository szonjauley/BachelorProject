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

def clean_data(data:pd.DataFrame, confidence:float=0.9) ->pd.DataFrame:
    """
    Takes the combined data, filters it for the specified confidence and successful frames and removes all non-AU_r columns
    """
    data = data[
        (data["confidence"] > confidence) &
        (data["success"] == 1)
    ]

    non_au_cols = ["speaker", "frame", "timestamp", "confidence", "success", 
                   "AU04_c", "AU12_c", "AU15_c", "AU23_c", "AU28_c", "AU45_c"]
    data = data.drop(columns=non_au_cols)
    data = data.set_index("person_ID")

    return data


def compute_person_stats(data: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the cleaned data and returns personal level stats indexed by person_ID
    """

    grouped = data.groupby(level="person_ID")

    q25 = lambda x: x.quantile(0.25)
    q25.__name__ = "q25"

    q75 = lambda x: x.quantile(0.75)
    q75.__name__ = "q75"

    stats = grouped.agg(["min", "max", "median", "mean", "std", q25, q75])

    # flatten columns
    stats.columns = [f"{au}_{metric}" for au, metric in stats.columns]

    return stats


def compute_group_stats(person_stats:pd.DataFrame) ->pd.DataFrame:
    """
    Takes person level stats and computes population-level statistics from person-level means
    Prevents participants with more frames from biasing results
    """

    # Keep only the mean columns
    mean_cols = [c for c in person_stats.columns if c.endswith("_mean")]

    all_stats = {}

    for col in mean_cols:
        base_name = col[:-5]  # strip "_mean"
        stats = {
            f"{base_name}_group_mean": person_stats[col].mean(),
            f"{base_name}_group_std": person_stats[col].std(),
            f"{base_name}_group_median": person_stats[col].median(),
            f"{base_name}_group_q25": person_stats[col].quantile(0.25),
            f"{base_name}_group_q75": person_stats[col].quantile(0.75),
            f"{base_name}_group_min": person_stats[col].min(),
            f"{base_name}_group_max": person_stats[col].max(),
        }

        all_stats.update(stats)

    group_stats = pd.DataFrame(all_stats, index=[0])

    return group_stats

def main(base_folder, speaker="all", confidence=0.9):

    print("Loading data...")
    combined = load_all_data(base_folder, speaker)

    print("Cleaning data...")
    cleaned = clean_data(combined, confidence)

    print("Computing person-level statistics...")
    person_stats = compute_person_stats(cleaned)

    print("Computing group-level statistics...")
    group_stats = compute_group_stats(person_stats)

    # Save
    combined_file = f"combined_AUs_labeled_{speaker}.csv"
    cleaned_file = f"cleaned_AUs_labeled_{speaker}_{confidence}.csv"
    person_file = "AU_stats_person_level.csv"
    group_file = "AU_stats_group_level.csv"

    combined.to_csv(combined_file, index=False)
    cleaned.to_csv(cleaned_file)
    person_stats.to_csv(person_file)
    group_stats.to_csv(group_file)

    print(f"\nSaved:")
    print(combined_file)
    print(cleaned_file)
    print(person_file)
    print(group_file)

    return combined, cleaned, person_stats, group_stats

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

