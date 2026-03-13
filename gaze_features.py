import pandas as pd
from pathlib import Path

def load_data(file:Path, depression:str="all") -> pd.DataFrame:
    """
    Load clean data and filter for depression status
    """
    df = pd.read_csv(file)

    if depression == "yes":
        df = df[df["depression"]==1]
    elif depression == "no":
        df = df[df["depression"]==0]

    return df