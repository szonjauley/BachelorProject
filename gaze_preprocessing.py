import pandas as pd
from pathlib import Path

def load_all_data(base_folder:Path, depression_file:Path) -> pd.DataFrame:
    """
    Loads all gaze csv files into one dataframe with a person_ID column.
    Takes folder path for all participants' data and the label file as input and returns dataframe with additional ID column and labels
    """
    base_path = Path(base_folder)
    dfs = []

    for person_folder in sorted(base_path.glob("*_P")):
        person_id = person_folder.name.split("_")[0]
        file_path = person_folder / f"{person_id}_CLNF_gaze_labeled.csv"

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

    depression_values = pd.read_csv(depression_file)

    labelmap = depression_values.set_index("Participant_ID")["PHQ8_Binary"]
    labelmap.index = labelmap.index.astype(str)
    data["depression"] = data["person_ID"].map(labelmap)

    return data

def clean_data(data:pd.DataFrame, confidence:float=0.9) ->pd.DataFrame:
    """
    Takes the combined data, filters it for the specified confidence and successful frames and removes all non-gaze columns
    Returns clean data indexed by person_ID including depression labels and segment type
    """
    data = data[
        (data["confidence"] > confidence) &
        (data["success"] == 1)
    ]

    non_gaze_cols = ["frame", "timestamp", "confidence", "success"]
    data = data.drop(columns=non_gaze_cols)
    data = data.set_index("person_ID")

    return data
