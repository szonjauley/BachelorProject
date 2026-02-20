import pandas as pd
import statsmodels.formula.api as smf

# -----------------------------
# PATHS (adjust these)
# -----------------------------

SP_DEP = "all_processed_data/speaking_depressed/AU_stats_wide_person_level_speaking_yes_0.7.csv"
SP_NON = "all_processed_data/speaking_non-depressed/AU_stats_wide_person_level_speaking_no_0.7.csv"
LI_DEP = "all_processed_data/listening_depressed/AU_stats_wide_person_level_listening_yes_0.7.csv"
LI_NON = "all_processed_data/listening_non-depressed/AU_stats_wide_person_level_listening_no_0.7.csv"

# -----------------------------
# LOAD
# -----------------------------

sp_dep = pd.read_csv(SP_DEP)
sp_non = pd.read_csv(SP_NON)
li_dep = pd.read_csv(LI_DEP)
li_non = pd.read_csv(LI_NON)

# Remove unnamed columns if needed
for df in [sp_dep, sp_non, li_dep, li_non]:
    df.drop(columns=df.columns[df.columns.str.contains("Unnamed")], inplace=True)

# Add labels
sp_dep["Depression"] = 1
sp_non["Depression"] = 0
li_dep["Depression"] = 1
li_non["Depression"] = 0

sp_dep["Role"] = 1   # speaking
sp_non["Role"] = 1
li_dep["Role"] = 0   # listening
li_non["Role"] = 0

# Combine everything
df = pd.concat([sp_dep, sp_non, li_dep, li_non], ignore_index=True)

# -----------------------------
# SELECT AUs
# -----------------------------

au_columns = [col for col in df.columns if "_mean" in col]

print("\nInteraction regression results:\n")

for au in au_columns:

    formula = f"{au} ~ Depression + Role + Depression:Role"
    model = smf.ols(formula, data=df).fit()

    interaction_p = model.pvalues.get("Depression:Role", None)

    print(f"{au}")
    print(f"  Interaction p-value: {interaction_p:.4f}")
    print(f"  R^2: {model.rsquared:.4f}")
    print("-" * 40)
