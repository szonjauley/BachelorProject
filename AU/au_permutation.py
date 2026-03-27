import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from pathlib import Path

# PATHS
SCRIPT_DIR = Path(__file__).parent.resolve() # BachelorProject/au
DATA_DIR = SCRIPT_DIR.parent / "data"  # BachelorProject/data
OUTPUT_DIR =SCRIPT_DIR.parent / "output" / "au" # BachelorProject/output/au
INPUT_PATH = DATA_DIR / "au_aggregation.csv" # BachelorProject/data/au_aggregation.csv

def prepare_data(file):
    data = pd.read_csv(file)
    data_filtered = data[
        ~(data["segment_type"] == "all") &
        ~data["AU"].str.endswith("_c")
    ]
    return data_filtered

def permutation_test_interaction(df, n_perm=5000):
    """
    Takes clean data and number of permutations
    Returns p-values for each AU score
    """

    results = []

    aus = df["AU"].unique()

    for au in aus:
        df_au = df[df["AU"] == au].copy()

        # Ensure categorical
        df_au["segment_type"] = df_au["segment_type"].astype("category")

        # --- Fit real model ---
        model = smf.ols("value ~ depressed * segment_type", data=df_au).fit()

        # Get correct interaction term name dynamically
        interaction_term = [t for t in model.params.index if "depressed:segment_type" in t][0]
        T_obs = model.tvalues[interaction_term]

        # --- Prepare subject-level labels ---
        subjects = df_au[["person_id", "depressed"]].drop_duplicates()

        T_perm = []

        for _ in range(n_perm):
            # Shuffle depression at subject level
            shuffled = subjects.copy()
            shuffled["depressed"] = np.random.permutation(shuffled["depressed"].values)

            # Merge back
            df_perm = df_au.drop(columns="depressed").merge(shuffled, on="person_id")

            # Fit model
            model_perm = smf.ols("value ~ depressed * segment_type", data=df_perm).fit()

            T_perm.append(model_perm.tvalues[interaction_term])

        T_perm = np.array(T_perm)

        # Two-sided p-value
        p_value = np.mean(np.abs(T_perm) >= np.abs(T_obs))

        results.append({
            "AU": au,
            "T_obs": T_obs,
            "p_value": p_value
        })

    results_df = pd.DataFrame(results)

    reject, p_bonf, _, _ = multipletests(results_df["p_value"], method="bonferroni")

    results_df["p_bonf"] = p_bonf
    results_df["significant"] = reject

    return results_df

def main(input_file, n_perm, output_folder):
    data = prepare_data(input_file)
    results = []

    for metric in ["mean", "std"]:
        df_metric = data[data["stat"] == metric].copy()
        res_df = permutation_test_interaction(df_metric, n_perm)
        res_df["stat"] = metric
        results.append(res_df)

    results_df = pd.concat(results, ignore_index=True)
    output_path = Path(output_folder) / "au_permutation.csv"
    results_df.to_csv(output_path, index=False)
    print("Saved:", output_path)

if __name__ == "__main__":

    main(
        input_file=INPUT_PATH,
        output_folder=OUTPUT_DIR,
        n_perm=5000
    )