import pandas as pd
from pathlib import Path

def load_all_data(base_folder:Path, speaker:str="all") -> pd.DataFrame:
    """
    Loads all AU csv files into one dataframe with a person_ID column.
    Takes folder path and speaker mode as input and returns dataframe with additional ID column
    """
    base_path = Path(base_folder)
    dfs = []

    for person_folder in sorted(base_path.glob("*_P")):
        person_id = person_folder.name.split("_")[0]
        file_path = person_folder / f"{person_id}_CLNF_AUs_labeled.csv"

        if not file_path.exists():
            print(f"Missing file for {person_id}")
            continue

        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()

        df["person_ID"] = person_id
        dfs.append(df)

    if not dfs:
        raise ValueError("No AU files found.")

    data = pd.concat(dfs, ignore_index=True)

    # Speaker mode filter 
    if speaker == "listening":
        data = data[data["speaker"] == "Listening"]
    elif speaker == "speaking":
        data = data[data["speaker"] == "Speaking"]

    return data