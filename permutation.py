import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

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

    return results