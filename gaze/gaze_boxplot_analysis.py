import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# PATHS
SCRIPT_DIR = Path(__file__).parent.resolve() # BachelorProject/gaze
DATA_DIR = SCRIPT_DIR.parent / "data"  # BachelorProject/data
OUTPUT_DIR =SCRIPT_DIR.parent / "output" / "gaze" / "boxplots" # BachelorProject/output/gaze/boxplots
INPUT_PATH = DATA_DIR / "gaze_aggregation.csv" # BachelorProject/data/gaze_aggregation.csv

statS = ["mean", "std"]

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def get_group(df, stat, segment_type, depressed=None):
    subset = df[(df["stat"] == stat) & (df["segment_type"] == segment_type)]
    if depressed is not None:
        subset = subset[subset["depressed"] == depressed]
    return subset.copy()

def plot_comparison(df, label_1, label_2, stat, filename,
                    segment_type_1, segment_type_2, depressed_1=None, depressed_2=None, y_lim=None):
    df_1 = get_group(df, stat, segment_type_1, depressed_1)
    df_2 = get_group(df, stat, segment_type_2, depressed_2)

    df_1["group"] = label_1
    df_2["group"] = label_2

    plot_df = pd.concat([df_1, df_2], ignore_index=True)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=plot_df, x="group", y="value")
    plt.ylabel("delta_deg")
    plt.xticks(rotation=20)
    plt.title(f"{label_1} vs {label_2} ({stat})")

    if y_lim is not None:
        plt.ylim(y_lim)

    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")

# -----------------------------
# RUN ALL COMPARISONS
# -----------------------------
df = pd.read_csv(INPUT_PATH)

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

for stat in statS:
    for label1, label2, seg1, seg2, dep1, dep2, base_name in comparisons:
        filename = f"{stat}_{base_name}.png"
        plot_comparison(
            df=df,
            label_1=label1,
            label_2=label2,
            stat=stat,
            filename=filename,
            segment_type_1=seg1,
            segment_type_2=seg2,
            depressed_1=dep1,
            depressed_2=dep2,
            y_lim=(0, 3),
        )

print("All plots generated successfully.")