import pandas as pd
from pathlib import Path
from scipy.stats import shapiro

def load_data(file:Path) -> pd.DataFrame:
    """
    Load aggregated data
    """
    df = pd.read_csv(file)

    return df

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
        df.groupby(["stat", "segment", "depression"])["delta_deg"]
        .apply(shapiro_test)
        .unstack()
        .reset_index()
    )

    # add "all depression" rows
    res_all = (
        df.groupby(["stat", "segment"])["delta_deg"]
        .apply(shapiro_test)
        .unstack()
        .reset_index()
    )
    res_all["depression"] = "all"

    # combine
    summary = pd.concat([res, res_all], ignore_index=True)

    summary = summary.rename(columns={
        "segment": "segment_type",
        "depression": "depressed"
    })

    summary = summary.sort_values(["stat", "segment_type", "depressed"])

    return summary