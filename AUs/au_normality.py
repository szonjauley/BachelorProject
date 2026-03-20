import pandas as pd
from pathlib import Path
from scipy.stats import shapiro


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