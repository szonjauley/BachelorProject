import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Output folder
OUTPUT_DIR = "boxplots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# PLOTTING FUNCTION
# -----------------------------

def plot_comparison(path_1, label_1, path_2, label_2, statistic, filename):
    
    df_1 = pd.read_csv(path_1)
    df_2 = pd.read_csv(path_2)
    
    df_1["group"] = label_1
    df_2["group"] = label_2
    
    df = pd.concat([df_1, df_2], ignore_index=True)
    
    # Select statistic columns
    stat_cols = [col for col in df.columns if col.endswith(f"_{statistic}")]
    
    df_stat = df[stat_cols + ["group"]]
    
    df_long = df_stat.melt(
        id_vars="group",
        var_name="AU",
        value_name="value"
    )
    
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=df_long, x="AU", y="value", hue="group")
    plt.xticks(rotation=45)
    plt.title(f"{label_1} vs {label_2} ({statistic})")
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Saved: {save_path}")


# -----------------------------
# RUN ALL COMPARISONS
# -----------------------------

comparisons = [

    # 1. All speaking vs all listening
    (SPEAKING_ALL_PATH, "Speaking (All)",
     LISTENING_ALL_PATH, "Listening (All)",
     "speaking_all_vs_listening_all.png"),

    # 2. Speaking non-depressed vs listening non-depressed
    (SPEAKING_NON_DEPRESSED_PATH, "Speaking Non-Depressed",
     LISTENING_NON_DEPRESSED_PATH, "Listening Non-Depressed",
     "speaking_non_dep_vs_listening_non_dep.png"),

    # 3. Speaking depressed vs listening depressed
    (SPEAKING_DEPRESSED_PATH, "Speaking Depressed",
     LISTENING_DEPRESSED_PATH, "Listening Depressed",
     "speaking_dep_vs_listening_dep_duplicate.png"),

    # 4. Listening non-depressed vs listening depressed
    (LISTENING_NON_DEPRESSED_PATH, "Listening Non-Depressed",
     LISTENING_DEPRESSED_PATH, "Listening Depressed",
     "listening_non_dep_vs_dep.png"),

    # 5. Speaking non-depressed vs speaking depressed
    (SPEAKING_NON_DEPRESSED_PATH, "Speaking Non-Depressed",
     SPEAKING_DEPRESSED_PATH, "Speaking Depressed",
     "speaking_non_dep_vs_dep.png"),
]

for path1, label1, path2, label2, filename in comparisons:
    plot_comparison(path1, label1, path2, label2, STATISTIC, filename)

print("All plots generated successfully.")