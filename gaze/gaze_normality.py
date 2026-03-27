import pandas as pd
from pathlib import Path
from scipy.stats import shapiro

# Directory where the current script is located
SCRIPT_DIR = Path(__file__).parent.resolve() # BachelorProject/gaze
DATA_DIR = SCRIPT_DIR.parent / "data" # BachelorProject/data
OUTPUT_DIR =SCRIPT_DIR.parent / "output" # BachelorProject/output
INPUT_PATH = DATA_DIR / "gaze_aggregation.csv" # BachelorProject/data/gaze_aggregation.csv
OUTPUT_PATH = OUTPUT_DIR / "gaze" / "gaze_normality.csv" # BachelorProject/output/gaze/gaze_normality.csv

def load_data(file:Path) -> pd.DataFrame:
    """
    Load aggregated data
    """
    return pd.read_csv(file)

def shapiro_test(x) -> pd.Series:
    """
    Run Shapiro normality test on a selected group
    """
    stat, p = shapiro(x)
    return pd.Series({
        "n": len(x),
        "shapiro_stat": stat,
        "p_value": p,
        "normal": p > 0.05
    })

def check_gaze_normality(df:pd.DataFrame):
    """
    Run Shapiro normality test on all data and format output
    """

    # normal groups
    res = (
        df.groupby(["stat", "segment_type", "depressed"])["value"]
        .apply(shapiro_test)
        .unstack()
        .reset_index()
    )

    # add "all depression" rows
    res_all = (
        df.groupby(["stat", "segment_type"])["value"]
        .apply(shapiro_test)
        .unstack()
        .reset_index()
    )
    res_all["depressed"] = "all"

    # combine
    summary = pd.concat([res, res_all], ignore_index=True)

    summary = summary.sort_values(["stat", "segment_type", "depressed"])

    return summary

def main(file:Path):
    df = load_data(file)
    gaze_summary = check_gaze_normality(df)
    gaze_summary.to_csv(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    main(
        file=INPUT_PATH
    )
