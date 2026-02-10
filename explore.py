import pandas as pd
import numpy as np

from pathlib import Path
import argparse

def get_files(base_folder):
    """
    Takes folder path and goes through the personal folders and finds the labeled AU csv files
    """
    base_path = Path(base_folder)

    for person_folder in sorted(base_path.glob("*_P")):
        person_id = person_folder.name.split("_")[0]
        file_path = person_folder / f"{person_id}_CLNF_AUs_labeled.csv"

        if file_path.exists():
            yield file_path
        else:
            print(f"Missing file for {person_id}")

def read_personal_file(file, speaker="all"):
    """
    Takes file path and speaker mode as input 
    Returns a pd.DataFrame with clean column headers filtered for the speaker mode and the person's ID
    """
    file = Path(file)
    person_ID = file.name.split("_")[0]

    au_data = pd.read_csv(file)
    au_data.columns = au_data.columns.str.strip()

    if speaker == "listening":
        au_data = au_data[au_data["speaker"] == "Listening"]
    elif speaker == "speaking":
        au_data = au_data[au_data["speaker"] == "Speaking"]

    return person_ID, au_data

def get_personal_average(data: pd.DataFrame, confidence: float = 0.7) -> pd.Series:
    """
    Takes a person's AU scores over the entire video, filters them for the specified confidence and averages over the accepted frames
    Returns an average for each AU score as well as the number of frames accepted
    """
    confident_data = data[
        (data["confidence"] > confidence) & 
        (data["success"] == 1)
    ]

    au_scores = confident_data.select_dtypes(include=np.number)

    au_scores = au_scores.drop(
        columns=["speaker", "frame", "timestamp", "confidence", "success"],
        errors="ignore"
    )

    personal_average = au_scores.mean()

    return personal_average

def main(base_folder, output_csv, speaker="all"):

    results = []

    for file in get_files(base_folder):

        person_ID, data = read_personal_file(file, speaker=speaker)
        personal_average = get_personal_average(data)

        personal_average["person_ID"] = person_ID
        results.append(personal_average)

    final_df = pd.DataFrame(results).set_index("person_ID")

    final_df.to_csv(output_csv)
    print(f"Saved results to {output_csv}")

    return final_df

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Aggregate AU scores per person.")

    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Path to the base folder containing person directories"
    )

    parser.add_argument(
        "--speaker",
        type=str,
        default="all",
        choices=["all", "listening", "speaking"],
        help="Filter AU data by speaker role"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV filename"
    )

    args = parser.parse_args()

    # Auto-name the file if user doesn't provide one
    output_file = args.output or f"personal_au_averages_{args.speaker.lower()}.csv"

    df = main(
        base_folder=args.folder,
        output_csv=output_file,
        speaker=args.speaker.lower()
    )

    

#run it like: python explore.py --base_folder(required) folder path --output_csv(optional) output_name.csv --speaker(optional) speaking
