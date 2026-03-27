import pandas as pd
from pathlib import Path
from scipy.stats import shapiro
import argparse


def load_data(file: Path) -> pd.DataFrame:
    """Load AU data"""
    return pd.read_csv(file)

def shapiro_test(x) -> pd.Series:
    """
    Run Shapiro normality test
    """
    stat, p = shapiro(x)
    return pd.Series({
        "n": len(x),
        "shapiro_stat": stat,
        "p_value": p,
        "normal": p > (0.05/14) #dividing by 14 because of multiple comparison amongst the 14 AU values
    })

def compute_normality(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Shapiro tests for AU data"""

    results = []

    aus = sorted(df["AU"].unique())
    stats_list = ["mean", "std"]
    segment_types = sorted(df["segment_type"].unique())
    depressed_values = sorted(df["depressed"].unique())

    for au in aus:
        for stat in stats_list:
            for seg in segment_types:

                # per depression group
                for dep in depressed_values:
                    subset = df[
                        (df["AU"] == au) &
                        (df["stat"] == stat) &
                        (df["segment_type"] == seg) &
                        (df["depressed"] == dep)
                    ]["value"].dropna()

                    res = shapiro_test(subset)
                    res_dict = res.to_dict()

                    results.append({
                        "AU": au,
                        "stat": stat,
                        "depressed": dep,
                        "segment_type": seg,
                        **res_dict
                    })

                # all depression combined
                subset_all = df[
                    (df["AU"] == au) &
                    (df["stat"] == stat) &
                    (df["segment_type"] == seg)
                ]["value"].dropna()

                res = shapiro_test(subset_all)
                res_dict = res.to_dict()

                results.append({
                    "AU": au,
                    "stat": stat,
                    "depressed": "all",
                    "segment_type": seg,
                    **res_dict
                })

    return pd.DataFrame(results)

def main(file: Path):
    df = load_data(file)
    normality_df = compute_normality(df)
    output_path = Path(__file__).parent.parent / "data" / "au_normality.csv"
    normality_df.to_csv(output_path, index=False)
    print("Saved:", output_path)

if __name__ == "__main__":
    INPUT_PATH = Path(__file__).parent.parent / "data" / "au_aggregation.csv"
    main(file=INPUT_PATH)