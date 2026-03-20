import pandas as pd
from pathlib import Path


def load_data(file: Path) -> pd.DataFrame:
    """Load AU data"""
    return pd.read_csv(file)