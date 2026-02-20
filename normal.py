import pandas as pd
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import scipy.stats as stats

data = pd.read_csv("/Users/szonja/Documents/ITU/Thesis/au_long.csv") ###change this but using Pietro's au_long.csv

results = []

# Filter only the stats we care about
data_filtered = data[
    data["stat"].isin(["mean", "std"]) &
    ~data["AU"].str.endswith("_c")
]

# Get unique AUs, stats, segment types, and depressed values
aus = sorted(data_filtered["AU"].unique())
stats_list = ["mean", "std"]
segment_types = sorted(data_filtered["segment_type"].unique())
depressed_values = sorted(data_filtered["depressed"].unique())

# Loop through AU × stat × segment_type
for au in aus:
    for stat in stats_list:
        for seg in segment_types:

            # First, loop over 0 and 1 depressed
            for dep in depressed_values:
                subset = data_filtered[
                    (data_filtered["AU"] == au) &
                    (data_filtered["stat"] == stat) &
                    (data_filtered["segment_type"] == seg) &
                    (data_filtered["depressed"] == dep)
                ]["value"].dropna()

                if len(subset) >= 3:
                    shapiro_stat, p = shapiro(subset)
                    normal = p > 0.05
                else:
                    shapiro_stat, p, normal = None, None, None

                results.append({
                    "AU": au,
                    "stat": stat,
                    "depressed": dep,
                    "segment_type": seg,
                    "n": len(subset),
                    "shapiro_stat": shapiro_stat,
                    "p_value": p,
                    "normal": normal
                })

            # Then, do "all" depressed (no filtering on 0/1)
            subset_all = data_filtered[
                (data_filtered["AU"] == au) &
                (data_filtered["stat"] == stat) &
                (data_filtered["segment_type"] == seg)
            ]["value"].dropna()

            if len(subset_all) >= 3:
                shapiro_stat, p = shapiro(subset_all)
                normal = p > 0.05
            else:
                shapiro_stat, p, normal = None, None, None

            results.append({
                "AU": au,
                "stat": stat,
                "depressed": "all",
                "segment_type": seg,
                "n": len(subset_all),
                "shapiro_stat": shapiro_stat,
                "p_value": p,
                "normal": normal
            })

# Convert results to DataFrame
normality_df = pd.DataFrame(results)

# Save CSV
normality_df.to_csv("/Users/szonja/Documents/ITU/Thesis/normality_segmented_extended.csv", index=False)

# QQ Plots
# Get unique depressed values and segment types
depressed_values = sorted(data_filtered["depressed"].unique())
segment_types = sorted(data_filtered["segment_type"].unique())

# Loop over each AU
for au, au_group in data_filtered.groupby("AU"):

    n_rows = 3  # 3 rows: top = first row, bottom = last row (can adjust)
    n_cols = 6  # 3 columns per stat: [0,1,all] for segment_types

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5*n_cols, 4*n_rows))

    # Flatten axes for easy indexing
    axes = axes.flatten()

    # stats list
    stats_list = ["mean", "std"]

    plot_idx = 0

    for stat_idx, stat in enumerate(stats_list):
        for dep in depressed_values + ["all"]:
            for seg in segment_types:
                # Determine subset
                if dep == "all":
                    subset = au_group[
                        (au_group["stat"] == stat) &
                        (au_group["segment_type"] == seg)
                    ]["value"].dropna()
                else:
                    subset = au_group[
                        (au_group["stat"] == stat) &
                        (au_group["segment_type"] == seg) &
                        (au_group["depressed"] == dep)
                    ]["value"].dropna()

                # Compute column and row for layout
                # Left 3 columns = mean, right 3 columns = std
                stat_offset = 0 if stat == "mean" else 3
                # For 3 rows: row 0 = first segment, row 1 = second segment, row 2 = last? adjust if more
                row = segment_types.index(seg)
                col = stat_offset + (depressed_values + ["all"]).index(dep)
                # Protect against exceeding axes
                if row * n_cols + col >= len(axes):
                    continue
                ax = axes[row * n_cols + col]

                if len(subset) >= 3:
                    stats.probplot(subset, dist="norm", plot=ax)
                    ax.set_title(f"{stat}, dep={dep}, seg={seg}")
                else:
                    ax.set_title(f"{stat}, dep={dep}, seg={seg}\n(not enough data)")

                ax.set_xlabel("")
                ax.set_ylabel("")

    # Remove unused axes
    for i in range(len(axes)):
        if axes[i].has_data() is False:
            fig.delaxes(axes[i])

    fig.suptitle(f"Q-Q Plots for {au}", fontsize=14)
    plt.tight_layout()

    # Save figure
    plt.savefig(f"/Users/szonja/Documents/ITU/Thesis/QQ_plots/QQ_{au}.png", dpi=300) ### change this to where you want plots