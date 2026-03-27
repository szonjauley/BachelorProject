import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path

# Directory where the current script is located
SCRIPT_DIR = Path(__file__).parent.resolve() # BachelorProject/gaze
DATA_DIR = SCRIPT_DIR.parent / "data"  # BachelorProject/data
OUTPUT_DIR =SCRIPT_DIR.parent / "output" # BachelorProject/output
INPUT_PATH = DATA_DIR / "gaze_aggregation.csv" # BachelorProject/data/gaze_aggregation.csv
OUTPUT_PATH = OUTPUT_DIR / "gaze" / "gaze_permutation.csv" # BachelorProject/output/gaze/gaze_permutation.csv

N_PERM = 5000

def prepare_data(file):
    """
    Takes file path and reads data 
    Returns data filtered listening/speaking segment types
    """
    data = pd.read_csv(file)

    data_filtered = data[
        (data["segment_type"] != "all")
    ].copy()

    return data_filtered

def permutation_test(df, n_perm):
    """
    Takes clean data and number of permutations
    Returns p-value
    """

    # Ensure categorical
    df["segment_type"] = df["segment_type"].astype("category")

    # --- Fit real model ---
    model = smf.ols("value ~ depressed * segment_type", data=df).fit()

    interaction_term = [
        t for t in model.params.index if "depressed:segment_type" in t
    ][0]

    T_obs = model.tvalues[interaction_term]

    # --- Subject-level labels ---
    subjects = df[["person_id", "depressed"]].drop_duplicates()

    T_perm = []

    for _ in range(n_perm):
        shuffled = subjects.copy()
        shuffled["depressed"] = np.random.permutation(shuffled["depressed"].values)

        df_perm = df.drop(columns="depressed").merge(
            shuffled, on="person_id"
        )

        model_perm = smf.ols(
            "value ~ depressed * segment_type", data=df_perm
        ).fit()

        T_perm.append(model_perm.tvalues[interaction_term])

    T_perm = np.array(T_perm)

    # Two-sided p-value
    p_value = np.mean(np.abs(T_perm) >= np.abs(T_obs))

    return pd.DataFrame([{
        "T_obs": T_obs,
        "p_value": p_value,
        "significant": p_value < 0.05
    }])

def main(file, n_perm):
    data = prepare_data(file)

    results = []

    for metric in ["mean", "std"]:
        df_metric = data[data["stat"] == metric].copy()

        res_df = permutation_test(df_metric, n_perm)

        # add stat column
        res_df["stat"] = metric

        results.append(res_df)

    # combine results
    results_df = pd.concat(results, ignore_index=True)

    results_df.to_csv(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    main(
        file=INPUT_PATH,
        n_perm=N_PERM
    )