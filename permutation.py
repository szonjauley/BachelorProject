import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from pathlib import Path
import argparse

def prepare_data(file, metric):
    """
    Takes file path and reads data 
    Returns data filtered for relevant metrics
    """

    data = pd.read_csv(file)
    data_filtered = data[
        (data["stat"]==metric) &
        ~(data["segment_type"]=="all") &
        ~data["AU"].str.endswith("_c")]
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

    reject, p_fdr, _, _ = multipletests(results_df["p_value"], method="fdr_bh")

    results_df["p_fdr"] = p_fdr
    results_df["significant"] = reject

    return results_df

def main(input_file, metric, n_perm, output_folder):
    data = prepare_data(input_file, metric)
    results_df = permutation_test_interaction(data, n_perm)
    results_df.to_csv(f"{output_folder}/permutation_results_{metric}.csv", index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run permutation test on person level aggregated mean AU scores"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to au_long.csv"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(Path.cwd()),
        help="Path to folder for output file with test results"
    )

    parser.add_argument(
        "--n_perm",
        type=int,
        default=5000,
        help="Number of permutations"
    )

    parser.add_argument(
    "--metric",
    type=str,
    default="mean",
    choices=["mean", "std"],
    help="Metric the permutation test should be run on"
    )

    args = parser.parse_args()

    main(
        input_file=args.input,
        output_folder=args.output,
        n_perm=args.n_perm,
        metric=args.metric
    )