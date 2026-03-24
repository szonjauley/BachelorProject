import pandas as pd
from pathlib import Path

CONF_THRESH = 0.7

# Directory where the current script is located
SCRIPT_DIR = Path(__file__).parent.resolve() # BachelorProject/gaze
DATA_DIR = SCRIPT_DIR.parent / "data" # BachelorProject/data
ROOT_DIR = DATA_DIR / "participant_folders" # BachelorProject/data/participant_folders
DEPRESSION_PATH = DATA_DIR / "depression.csv" # BachelorProject/data/depression.csv
COMBINED_PATH = DATA_DIR / "gaze_combined_labeled.csv" # BachelorProject/data/gaze_combined_labeled.csv
CLEANED_PATH = DATA_DIR / f"gaze_cleaned_labeled_{CONF_THRESH}.csv" # BachelorProject/data/gaze_cleaned_labeled_0.7.csv

def load_all_data(base_folder:Path, depression_file:Path) -> pd.DataFrame:
    """
    Loads all gaze csv files into one dataframe with a person_id column.
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

        df["person_id"] = person_id
        dfs.append(df)

    if not dfs:
        raise ValueError("No gaze files found.")

    data = pd.concat(dfs, ignore_index=True)

    depression_values = pd.read_csv(depression_file)

    labelmap = depression_values.set_index("Participant_ID")["PHQ8_Binary"]
    labelmap.index = labelmap.index.astype(str)
    data["depressed"] = data["person_id"].map(labelmap)

    return data

def clean_data(data:pd.DataFrame, confidence:float) ->pd.DataFrame:
    """
    Takes the combined data, filters it for the specified confidence and successful frames and removes all non-gaze columns
    Returns clean data indexed by person_id including depression labels and segment type
    """
    data = data[
        (data["confidence"] > confidence) &
        (data["success"] == 1)
    ]

    non_gaze_cols = ["frame", "timestamp", "confidence", "success"]
    data = data.drop(columns=non_gaze_cols)
    data = data.set_index("person_id")

    return data

def main(base_folder:Path, depression_file:Path, confidence:float):

    print("Loading data...")
    combined = load_all_data(base_folder, depression_file)

    print("Cleaning data...")
    cleaned = clean_data(combined, confidence)

    print(f"\nSaved:")
    combined.to_csv(COMBINED_PATH, index=False)
    print(COMBINED_PATH)
    cleaned.to_csv(CLEANED_PATH)
    print(CLEANED_PATH)

    return

if __name__ == "__main__":

    main(
        base_folder=ROOT_DIR,
        depression_file=DEPRESSION_PATH,
        confidence=CONF_THRESH
    )
