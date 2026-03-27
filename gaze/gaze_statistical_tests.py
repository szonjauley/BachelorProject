import os
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
from pathlib import Path

# PATHS
SCRIPT_DIR = Path(__file__).parent.resolve() # BachelorProject/gaze
DATA_DIR = SCRIPT_DIR.parent / "data"  # BachelorProject/data
OUTPUT_DIR =SCRIPT_DIR.parent / "output" / "gaze" # BachelorProject/output/gaze
INPUT_PATH = DATA_DIR / "gaze_aggregation.csv" # BachelorProject/data/gaze_aggregation.csv

STATISTICS = ["mean", "std"]
ALPHA = 0.05

def get_group(df, stat, segment_type, depressed=None):
    subset = df[(df["stat"] == stat) & (df["segment_type"] == segment_type)]
    if depressed is not None:
        subset = subset[subset["depressed"] == depressed]
    return subset

def paired_test(df1, df2):
    merged = pd.merge(
        df1[["person_id", "value"]],
        df2[["person_id", "value"]],
        on="person_id",
        suffixes=("_1", "_2")
    )
    stat, p = wilcoxon(merged["value_1"], merged["value_2"])
    return stat, p, len(merged)

def independent_test(df1, df2):
    x = df1["value"]
    y = df2["value"]
    stat, p = mannwhitneyu(x, y, alternative="two-sided")
    return stat, p, len(x), len(y)

def run_tests(df, stat_name):
    results = []

    # 1) Speaking vs Listening (All persons) — paired
    g1 = get_group(df, stat_name, "speaking")
    g2 = get_group(df, stat_name, "listening")
    stat, p, n = paired_test(g1, g2)
    results.append({
        "comparison": "speaking_all_vs_listening_all",
        "test": "Wilcoxon",
        "n": n,
        "statistic": stat,
        "p_value": p,
        "significant": p <= ALPHA
    })

    # 2) Speaking non-depressed vs Listening non-depressed — paired
    g1 = get_group(df, stat_name, "speaking", 0)
    g2 = get_group(df, stat_name, "listening", 0)
    stat, p, n = paired_test(g1, g2)
    results.append({
        "comparison": "speaking_non_dep_vs_listening_non_dep",
        "test": "Wilcoxon",
        "n": n,
        "statistic": stat,
        "p_value": p,
        "significant": p <= ALPHA
    })

    # 3) Speaking depressed vs Listening depressed — paired
    g1 = get_group(df, stat_name, "speaking", 1)
    g2 = get_group(df, stat_name, "listening", 1)
    stat, p, n = paired_test(g1, g2)
    results.append({
        "comparison": "speaking_dep_vs_listening_dep",
        "test": "Wilcoxon",
        "n": n,
        "statistic": stat,
        "p_value": p,
        "significant": p <= ALPHA
    })

    # 4) Listening non-depressed vs Listening depressed — independent
    g1 = get_group(df, stat_name, "listening", 0)
    g2 = get_group(df, stat_name, "listening", 1)
    stat, p, n1, n2 = independent_test(g1, g2)
    results.append({
        "comparison": "listening_non_dep_vs_dep",
        "test": "Mann-Whitney U",
        "n1": n1,
        "n2": n2,
        "statistic": stat,
        "p_value": p,
        "significant": p <= ALPHA
    })

    # 5) Speaking non-depressed vs Speaking depressed — independent
    g1 = get_group(df, stat_name, "speaking", 0)
    g2 = get_group(df, stat_name, "speaking", 1)
    stat, p, n1, n2 = independent_test(g1, g2)
    results.append({
        "comparison": "speaking_non_dep_vs_dep",
        "test": "Mann-Whitney U",
        "n1": n1,
        "n2": n2,
        "statistic": stat,
        "p_value": p,
        "significant": p <= ALPHA
    })

    return pd.DataFrame(results)

def main():
    df = pd.read_csv(INPUT_PATH)

    for stat in STATISTICS:
        results_df = run_tests(df, stat)
        output_path = os.path.join(OUTPUT_DIR, f"gaze_stat_test_{stat}.csv")
        results_df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
        print(results_df.to_string(index=False))
        print("-" * 60)

if __name__ == "__main__":
    main()