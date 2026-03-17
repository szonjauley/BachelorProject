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