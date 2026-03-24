import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

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