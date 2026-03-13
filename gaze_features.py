import pandas as pd
import numpy as np
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

def average_eyes(df:pd.DataFrame) -> pd.DataFrame:
    """
    Take average gaze of the two eyes and renormalize vector
    """
    df["x_avg"] = (df.x_0 + df.x_1)/2
    df["y_avg"] = (df.y_0 + df.y_1)/2
    df["z_avg"] = (df.z_0 + df.z_1)/2

    # renormalize the averaged gaze vector
    norm = np.sqrt(df.x_avg**2 + df.y_avg**2 + df.z_avg**2)
    df[["x_avg","y_avg","z_avg"]] = df[["x_avg","y_avg","z_avg"]].div(norm, axis=0)

    return df