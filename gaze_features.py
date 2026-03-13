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

def get_delta(df:pd.DataFrame, group_cols:list) -> pd.DataFrame:
    """
    Computes delta angle between frames and returns it in radians and degrees
    """
    V = df[["x_avg","y_avg","z_avg"]].to_numpy()

    V_prev = (
        df.groupby(group_cols)[["x_avg","y_avg","z_avg"]]
        .shift()
        .to_numpy()
    )

    dot = np.einsum("ij,ij->i", V, V_prev)

    df["delta_rad"] = np.arccos(np.clip(dot, -1, 1))
    df["delta_deg"] = np.rad2deg(df["delta_rad"])

    return df

def main(file:Path, depression:str):

    print("Loading data...")
    df = load_data(file, depression)

    print("Averaging eyes...")
    df = average_eyes(df)

    
    # --- ALL ---
    print("Creating file for all...")
    df_all = df.copy()
    df_all["segment"] = (df_all["person_ID"] != df_all["person_ID"].shift()).cumsum()
    df_all = get_delta(df_all, ["segment"])

    # --- LISTENING ---
    print("Creating file for listening...")
    df_listening = df.copy()
    boundary = (
        (df_listening["person_ID"] != df_listening["person_ID"].shift()) |
        (df_listening["speaker"] != df_listening["speaker"].shift())
    )
    df_listening["segment"] = boundary.cumsum()
    df_listening = df_listening[df_listening["speaker"] == "Listening"]
    df_listening = get_delta(df_listening, ["segment"])

    # --- SPEAKING ---
    print("Creating file for sepaking...")
    df_speaking = df.copy()
    boundary = (
        (df_speaking["person_ID"] != df_speaking["person_ID"].shift()) |
        (df_speaking["speaker"] != df_speaking["speaker"].shift())
    )
    df_speaking["segment"] = boundary.cumsum()
    df_speaking = df_speaking[df_speaking["speaker"] == "Speaking"]
    df_speaking = get_delta(df_speaking, ["segment"])

    print("Saving files...")
    combined_file = f"combined_gaze_deltas_{depression}.csv"
    listening_file = f"listening_gaze_deltas_{depression}.csv"
    speaking_file = f"speaking_gaze_deltas_{depression}.csv"

    df_all.to_csv(combined_file, index=False)
    df_listening.to_csv(listening_file, index=False)
    df_speaking.to_csv(speaking_file, index=False)

    return