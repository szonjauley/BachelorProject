import os
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

# -----------------------------
# CONFIGURATION
# -----------------------------

# File paths
SPEAKING_DEPRESSED_PATH = "all_processed_data/speaking_depressed/AU_stats_wide_person_level_speaking_yes_0.7.csv"
SPEAKING_NON_DEPRESSED_PATH = "all_processed_data/speaking_non-depressed/AU_stats_wide_person_level_speaking_no_0.7.csv"

LISTENING_DEPRESSED_PATH = "all_processed_data/listening_depressed/AU_stats_wide_person_level_listening_yes_0.7.csv"
LISTENING_NON_DEPRESSED_PATH = "all_processed_data/listening_non-depressed/AU_stats_wide_person_level_listening_no_0.7.csv"

SPEAKING_ALL_PATH = "all_processed_data/speaking_all/AU_stats_wide_person_level_speaking_all_0.7.csv"
LISTENING_ALL_PATH = "all_processed_data/listening_all/AU_stats_wide_person_level_listening_all_0.7.csv"

# Statistic to analyze: "mean", "std", or "median"
STATISTIC = "mean"

OUTPUT_DIR = "statistical_tests"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# STATISTICAL TEST FUNCTION
# -----------------------------

def run_statistical_test(path_1, label_1, path_2, label_2, statistic, filename):

    df_1 = pd.read_csv(path_1)
    df_2 = pd.read_csv(path_2)

    # Select statistic columns
    stat_cols_1 = [col for col in df_1.columns if col.endswith(f"_{statistic}")]
    stat_cols_2 = [col for col in df_2.columns if col.endswith(f"_{statistic}")]

    # Ensure same AU order
    stat_cols = sorted(list(set(stat_cols_1).intersection(stat_cols_2)))

    results = []

    # Determine if paired comparison
    paired = (
        ("Speaking" in label_1 and "Listening" in label_2) or
        ("Listening" in label_1 and "Speaking" in label_2)
    )

    for col in stat_cols:

        x = df_1[col].dropna()
        y = df_2[col].dropna()

        if paired:
            # Must align by subject index for paired test
            min_len = min(len(x), len(y))
            stat, p = wilcoxon(x.iloc[:min_len], y.iloc[:min_len])
            test_name = "Wilcoxon_signed_rank"
        else:
            stat, p = mannwhitneyu(x, y, alternative="two-sided")
            test_name = "Mann_Whitney_U"

        results.append({
            "AU": col,
            "test": test_name,
            "statistic": stat,
            "p_value": p,
            "accept_h_0": p >= 0.05
        })

    results_df = pd.DataFrame(results)

    save_path = os.path.join(OUTPUT_DIR, filename)
    results_df.to_csv(save_path, index=False)

    print(f"Saved results to {save_path}")


# -----------------------------
# RUN ALL COMPARISONS
# -----------------------------

comparisons = [

    # 1. All speaking vs all listening (paired)
    (SPEAKING_ALL_PATH, "Speaking (All)",
     LISTENING_ALL_PATH, "Listening (All)",
     "speaking_all_vs_listening_all_stats_{STATISTIC}.csv"),

    # 2. Speaking non-depressed vs listening non-depressed (paired)
    (SPEAKING_NON_DEPRESSED_PATH, "Speaking Non-Depressed",
     LISTENING_NON_DEPRESSED_PATH, "Listening Non-Depressed",
     "speaking_non_dep_vs_listening_non_dep_stats_{STATISTIC}.csv"),

    # 3. Speaking depressed vs listening depressed (paired)
    (SPEAKING_DEPRESSED_PATH, "Speaking Depressed",
     LISTENING_DEPRESSED_PATH, "Listening Depressed",
     "speaking_dep_vs_listening_dep_stats_{STATISTIC}.csv"),

    # 4. Listening non-depressed vs listening depressed (independent)
    (LISTENING_NON_DEPRESSED_PATH, "Listening Non-Depressed",
     LISTENING_DEPRESSED_PATH, "Listening Depressed",
     "listening_non_dep_vs_dep_stats_{STATISTIC}.csv"),

    # 5. Speaking non-depressed vs speaking depressed (independent)
    (SPEAKING_NON_DEPRESSED_PATH, "Speaking Non-Depressed",
     SPEAKING_DEPRESSED_PATH, "Speaking Depressed",
     "speaking_non_dep_vs_dep_stats_{STATISTIC}.csv"),
]

for path1, label1, path2, label2, filename in comparisons:
    run_statistical_test(path1, label1, path2, label2, STATISTIC, filename)

print("All statistical tests completed.")