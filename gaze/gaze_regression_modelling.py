import os
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path

# ======================================================
# PATHS
# ======================================================
SCRIPT_DIR = Path(__file__).parent.resolve() # BachelorProject/gaze
DATA_DIR = SCRIPT_DIR.parent / "data"  # BachelorProject/data
OUTPUT_DIR =SCRIPT_DIR.parent / "output" / "gaze" # BachelorProject/output/gaze
INPUT_PATH = DATA_DIR / "gaze_aggregation.csv" # BachelorProject/data/gaze_aggregation.csv

STATISTICS = ["mean", "std"]

# ======================================================
# LOAD DATA
# ======================================================
df = pd.read_csv(INPUT_PATH)

# Keep only speaking and listening rows for the interaction model
df = df[df["segment"].isin(["speaking", "listening"])].copy()

# Recode predictors
df["Depression"] = df["depression"].astype(int)
df["Role"] = df["segment"].map({"listening": 0, "speaking": 1}).astype(int)

# ======================================================
# RUN REGRESSIONS
# ======================================================
for stat in STATISTICS:
    df_stat = df[df["stat"] == stat].copy()

    print(f"\nInteraction regression results ({stat}):\n")

    formula = "delta_deg ~ Depression + Role + Depression:Role"
    model = smf.ols(formula, data=df_stat).fit()

    interaction_p = model.pvalues.get("Depression:Role", None)

    print(f"Interaction p-value: {interaction_p:.4f}")
    print(f"R^2: {model.rsquared:.4f}")

    if interaction_p > 0.05:
        print("No evidence that the speaking-listening difference depends on depression status.")
    else:
        print("Evidence that the speaking-listening difference depends on depression status.")

    print("-" * 100)

    results = pd.DataFrame([{
        "stat": stat,
        "interaction_p": interaction_p,
        "r_squared": model.rsquared,
        "depression_p": model.pvalues.get("Depression", None),
        "role_p": model.pvalues.get("Role", None),
        "n": int(model.nobs),
    }])

    out_path = os.path.join(OUTPUT_DIR, f"gaze_interaction_regression_{stat}.csv")
    results.to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")