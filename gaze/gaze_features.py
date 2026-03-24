import pandas as pd
import numpy as np
from pathlib import Path
import argparse

# Directory where the current script is located
SCRIPT_DIR = Path(__file__).parent.resolve() # BachelorProject/gaze
DATA_DIR = SCRIPT_DIR.parent / "data" # BachelorProject/data
CLEANED_PATH = DATA_DIR / "gaze_cleaned_labeled_0.7.csv" # BachelorProject/data/gaze_cleaned_labeled_0.7.csv

def load_data(file:Path) -> pd.DataFrame:
    """
    Load clean data 
    """

    return pd.read_csv(file)

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

def main(file:Path):

    print("Loading data...")
    df = load_data(file)

    print("Averaging eyes...")
    df = average_eyes(df)

    
    # --- ALL ---
    print("Creating file for all...")
    df_all = df.copy()
    df_all["segment_type"] = (df_all["person_id"] != df_all["person_id"].shift()).cumsum()
    df_all = get_delta(df_all, ["segment_type"])

    # --- LISTENING ---
    print("Creating file for listening...")
    df_listening = df.copy()
    boundary = (
        (df_listening["person_id"] != df_listening["person_id"].shift()) |
        (df_listening["speaker"] != df_listening["speaker"].shift())
    )
    df_listening["segment_type"] = boundary.cumsum()
    df_listening = df_listening[df_listening["speaker"] == "Listening"]
    df_listening = get_delta(df_listening, ["segment_type"])

    # --- SPEAKING ---
    print("Creating file for sepaking...")
    df_speaking = df.copy()
    boundary = (
        (df_speaking["person_id"] != df_speaking["person_id"].shift()) |
        (df_speaking["speaker"] != df_speaking["speaker"].shift())
    )
    df_speaking["segment_type"] = boundary.cumsum()
    df_speaking = df_speaking[df_speaking["speaker"] == "Speaking"]
    df_speaking = get_delta(df_speaking, ["segment_type"])

    print("Saving files...")
    combined_file = DATA_DIR / "combined_gaze_deltas.csv"
    listening_file = DATA_DIR / "listening_gaze_deltas.csv"
    speaking_file = DATA_DIR / "speaking_gaze_deltas.csv"

    df_all.to_csv(combined_file, index=False)
    df_listening.to_csv(listening_file, index=False)
    df_speaking.to_csv(speaking_file, index=False)

    return

if __name__ == "__main__":

    main(
        file=CLEANED_PATH
    )