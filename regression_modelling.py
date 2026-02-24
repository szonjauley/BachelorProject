import argparse
from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf

# Path before the "all_processed_data" folder
parser = argparse.ArgumentParser()
parser.add_argument("base_path", help="Path that contains all_processed_data")
args = parser.parse_args()
data_root = Path(args.base_path) / "all_processed_data"

# File paths
SP_DEP = data_root / "speaking_depressed" / "AU_stats_wide_person_level_speaking_yes_0.7.csv"
SP_NON = data_root / "speaking_non-depressed" / "AU_stats_wide_person_level_speaking_no_0.7.csv"
LI_DEP = data_root / "listening_depressed" / "AU_stats_wide_person_level_listening_yes_0.7.csv"
LI_NON = data_root / "listening_non-depressed" / "AU_stats_wide_person_level_listening_no_0.7.csv"

# Load the files
sp_dep = pd.read_csv(SP_DEP)
sp_non = pd.read_csv(SP_NON)
li_dep = pd.read_csv(LI_DEP)
li_non = pd.read_csv(LI_NON)

# Remove unnamed columns if needed
for df in [sp_dep, sp_non, li_dep, li_non]:
    df.drop(columns=df.columns[df.columns.str.contains("Unnamed")], inplace=True)

# Add labels for depressed/non-depressed and speaking/listening
sp_dep["Depression"] = 1 # depressed
sp_non["Depression"] = 0 # non-depressed
li_dep["Depression"] = 1
li_non["Depression"] = 0

sp_dep["Role"] = 1 # speaking
sp_non["Role"] = 1
li_dep["Role"] = 0 # listening
li_non["Role"] = 0

# Combine everything
df = pd.concat([sp_dep, sp_non, li_dep, li_non], ignore_index=True)

# Select AUs
au_columns = [c for c in df.columns if c.startswith("AU") and c.endswith("_mean")]

print("\nInteraction regression results:\n")

for au in au_columns:

    formula = f"{au} ~ Depression + Role + Depression:Role"
    model = smf.ols(formula, data=df).fit()

    interaction_p = model.pvalues.get("Depression:Role", None)

    print(f"{au}")
    print(f"Interaction p-value: {interaction_p:.4f}")
    print(f"R^2: {model.rsquared:.4f}")

    if interaction_p > 0.05:
        print("No evidence that the speaking–listening difference depends on depression status.")
    else:
        print("Evidence that the speaking–listening difference depends on depression status.")

    print("-" * 100)