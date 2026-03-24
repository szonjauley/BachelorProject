import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# CONFIGURATION
# -----------------------------
INPUT_FILE = "gaze_aggregation.csv"
OUTPUT_DIR = "boxplots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

STATISTICS = ["mean", "std"]

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def get_group(df, statistic, segment, depressed=None):
    subset = df[(df["stat"] == statistic) & (df["segment"] == segment)]
    if depressed is not None:
        subset = subset[subset["depression"] == depressed]
    return subset.copy()

def plot_comparison(df, label_1, label_2, statistic, filename,
                    segment_1, segment_2, depressed_1=None, depressed_2=None):
    df_1 = get_group(df, statistic, segment_1, depressed_1)
    df_2 = get_group(df, statistic, segment_2, depressed_2)

    df_1["group"] = label_1
    df_2["group"] = label_2

    plot_df = pd.concat([df_1, df_2], ignore_index=True)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=plot_df, x="group", y="delta_deg")
    plt.xticks(rotation=20)
    plt.title(f"{label_1} vs {label_2} ({statistic})")
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")

# -----------------------------
# RUN ALL COMPARISONS
# -----------------------------
df = pd.read_csv(INPUT_FILE)

comparisons = [
    # 1. All speaking vs all listening
    ("Speaking (All)", "Listening (All)", "speaking", "listening", None, None, "speaking_all_vs_listening_all"),

    # 2. Speaking non-depressed vs listening non-depressed
    ("Speaking Non-Depressed", "Listening Non-Depressed", "speaking", "listening", 0, 0, "speaking_non_dep_vs_listening_non_dep"),

    # 3. Speaking depressed vs listening depressed
    ("Speaking Depressed", "Listening Depressed", "speaking", "listening", 1, 1, "speaking_dep_vs_listening_dep"),

    # 4. Listening non-depressed vs listening depressed
    ("Listening Non-Depressed", "Listening Depressed", "listening", "listening", 0, 1, "listening_non_dep_vs_dep"),

    # 5. Speaking non-depressed vs speaking depressed
    ("Speaking Non-Depressed", "Speaking Depressed", "speaking", "speaking", 0, 1, "speaking_non_dep_vs_dep"),
]

for statistic in STATISTICS:
    for label1, label2, seg1, seg2, dep1, dep2, base_name in comparisons:
        filename = f"{statistic}_{base_name}.png"
        plot_comparison(
            df=df,
            label_1=label1,
            label_2=label2,
            statistic=statistic,
            filename=filename,
            segment_1=seg1,
            segment_2=seg2,
            depressed_1=dep1,
            depressed_2=dep2
        )

print("All plots generated successfully.")