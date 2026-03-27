import os
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

# -----------------------------
# PATHS
# -----------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR.parent / "data"
OUTPUT_DIR = SCRIPT_DIR.parent / "output" / "AU" / "statistical_tests"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "au_aggregation.csv"

# -----------------------------
# CONFIGURATION
# -----------------------------
STATISTICS = ["mean", "std"]
ALPHA = 0.05

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def get_group(df, stat, segment_type, depressed=None):
    subset = df[(df["stat"] == stat) & (df["segment_type"] == segment_type)]
    if depressed is not None:
        subset = subset[subset["depressed"] == depressed]
    return subset


def run_statistical_test(df, label_1, label_2, statistic, filename):
    results = []

    paired = (
        ("Speaking" in label_1 and "Listening" in label_2) or
        ("Listening" in label_1 and "Speaking" in label_2)
    )

    # Use all AUs present in both groups
    aus_1 = set(df["AU"].unique())
    aus_2 = set(df["AU"].unique())
    aus = sorted(list(aus_1.intersection(aus_2)))

    for au in aus:
        df_au = df[df["AU"] == au]

        if "Speaking (All)" == label_1:
            g1 = get_group(df_au, statistic, "speaking")
        elif "Speaking Non-Depressed" == label_1:
            g1 = get_group(df_au, statistic, "speaking", 0)
        elif "Speaking Depressed" == label_1:
            g1 = get_group(df_au, statistic, "speaking", 1)
        elif "Listening (All)" == label_1:
            g1 = get_group(df_au, statistic, "listening")
        elif "Listening Non-Depressed" == label_1:
            g1 = get_group(df_au, statistic, "listening", 0)
        elif "Listening Depressed" == label_1:
            g1 = get_group(df_au, statistic, "listening", 1)
        else:
            raise ValueError(f"Unknown label_1: {label_1}")

        if "Speaking (All)" == label_2:
            g2 = get_group(df_au, statistic, "speaking")
        elif "Speaking Non-Depressed" == label_2:
            g2 = get_group(df_au, statistic, "speaking", 0)
        elif "Speaking Depressed" == label_2:
            g2 = get_group(df_au, statistic, "speaking", 1)
        elif "Listening (All)" == label_2:
            g2 = get_group(df_au, statistic, "listening")
        elif "Listening Non-Depressed" == label_2:
            g2 = get_group(df_au, statistic, "listening", 0)
        elif "Listening Depressed" == label_2:
            g2 = get_group(df_au, statistic, "listening", 1)
        else:
            raise ValueError(f"Unknown label_2: {label_2}")

        if paired:
            merged = pd.merge(
                g1[["person_id", "value"]],
                g2[["person_id", "value"]],
                on="person_id",
                suffixes=("_1", "_2")
            )

            stat, p = wilcoxon(merged["value_1"], merged["value_2"])
            test_name = "Wilcoxon_signed_rank"
        else:
            x = g1["value"]
            y = g2["value"]
            stat, p = mannwhitneyu(x, y, alternative="two-sided")
            test_name = "Mann_Whitney_U"

        results.append({
            "AU": au,
            "test": test_name,
            "statistic": stat,
            "p_value": p,
            f"Significant (p <= {ALPHA})": p <= ALPHA
        })

    results_df = pd.DataFrame(results)
    save_path = os.path.join(OUTPUT_DIR, filename)
    results_df.to_csv(save_path, index=False)
    print(f"Saved results to {save_path}")


# -----------------------------
# RUN ALL COMPARISONS
# -----------------------------

df = pd.read_csv(INPUT_FILE)

comparisons = [
    # 1. All speaking vs all listening (paired)
    ("Speaking (All)", "Listening (All)", "speaking_all_vs_listening_all"),

    # 2. Speaking non-depressed vs listening non-depressed (paired)
    ("Speaking Non-Depressed", "Listening Non-Depressed", "speaking_non_dep_vs_listening_non_dep"),

    # 3. Speaking depressed vs listening depressed (paired)
    ("Speaking Depressed", "Listening Depressed", "speaking_dep_vs_listening_dep"),

    # 4. Listening non-depressed vs listening depressed (independent)
    ("Listening Non-Depressed", "Listening Depressed", "listening_non_dep_vs_dep"),

    # 5. Speaking non-depressed vs speaking depressed (independent)
    ("Speaking Non-Depressed", "Speaking Depressed", "speaking_non_dep_vs_dep"),
]

for statistic in STATISTICS:
    for label1, label2, base_filename in comparisons:
        filename = f"{base_filename}_stats_{statistic}.csv"
        run_statistical_test(df, label1, label2, statistic, filename)

print("All statistical tests completed.")