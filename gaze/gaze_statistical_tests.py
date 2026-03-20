import os
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

INPUT_FILE = "gaze_aggregation.csv"
OUTPUT_DIR = "statistical_tests_long_minimal"
os.makedirs(OUTPUT_DIR, exist_ok=True)

STATISTICS = ["mean", "std"]
ALPHA = 0.05

def get_group(df, stat, segment, depression=None):
    subset = df[(df["stat"] == stat) & (df["segment"] == segment)]
    if depression is not None:
        subset = subset[subset["depression"] == depression]
    return subset

def paired_test(df1, df2):
    merged = pd.merge(
        df1[["person_ID", "delta_deg"]],
        df2[["person_ID", "delta_deg"]],
        on="person_ID",
        suffixes=("_1", "_2")
    )
    stat, p = wilcoxon(merged["delta_deg_1"], merged["delta_deg_2"])
    return stat, p, len(merged)

def independent_test(df1, df2):
    x = df1["delta_deg"]
    y = df2["delta_deg"]
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
    df = pd.read_csv(INPUT_FILE)

    for stat in STATISTICS:
        results_df = run_tests(df, stat)
        output_path = os.path.join(OUTPUT_DIR, f"results_{stat}.csv")
        results_df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
        print(results_df.to_string(index=False))
        print("-" * 60)

if __name__ == "__main__":
    main()