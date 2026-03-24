import os
from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf

# ======================================================
# CONFIGURATION
# ======================================================
INPUT_FILE = "au_long.csv"
OUTPUT_DIR = "interaction_regression_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

STATISTICS = ["mean", "std"]

# ======================================================
# LOAD DATA
# ======================================================
df = pd.read_csv(INPUT_FILE)

# Keep only speaking and listening rows for the interaction model
df = df[df["segment_type"].isin(["speaking", "listening"])].copy()

# Recode predictors
df["Depression"] = df["depressed"].astype(int)
df["Role"] = df["segment_type"].map({"listening": 0, "speaking": 1}).astype(int)

# ======================================================
# RUN REGRESSIONS
# ======================================================
for stat in STATISTICS:
    df_stat = df[df["stat"] == stat].copy()

    results = []

    print(f"\nInteraction regression results ({stat}):\n")

    for au in sorted(df_stat["AU"].dropna().unique()):
        df_au = df_stat[df_stat["AU"] == au].copy()

        formula = "value ~ Depression + Role + Depression:Role"
        model = smf.ols(formula, data=df_au).fit()

        interaction_p = model.pvalues.get("Depression:Role", None)

        print(f"{au}")
        print(f"Interaction p-value: {interaction_p:.4f}")
        print(f"R^2: {model.rsquared:.4f}")

        if interaction_p > 0.05:
            print("No evidence that the speaking-listening difference depends on depression status.")
        else:
            print("Evidence that the speaking-listening difference depends on depression status.")

        print("-" * 100)

        results.append({
            "AU": au,
            "stat": stat,
            "interaction_p": interaction_p,
            "r_squared": model.rsquared,
            "depression_p": model.pvalues.get("Depression", None),
            "role_p": model.pvalues.get("Role", None),
            "n": int(model.nobs),
        })

    out_path = os.path.join(OUTPUT_DIR, f"interaction_regression_results_{stat}.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")

