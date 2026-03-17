import pandas as pd
from pathlib import Path

def load_data(file:Path) -> pd.DataFrame:
    """
    Load aggregated data
    """
    df = pd.read_csv(file)

    return df