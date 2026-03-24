import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os

# Load data
CSV_PATH = "cleaned_gaze_labeled_0.7.csv"
df = pd.read_csv(CSV_PATH)

# Compute global limits across ALL person IDs
all_x = pd.concat([df["x_0"], df["x_1"], df["x_h0"], df["x_h1"]])
all_y = pd.concat([df["y_0"], df["y_1"], df["y_h0"], df["y_h1"]])
global_lim = max(abs(all_x.min()), abs(all_x.max()), abs(all_y.min()), abs(all_y.max()))

# Mean of two landmarks
def mid(a, b):
    return (a + b) / 2

# Output directory
out_dir = "gaze_scatter_plots"
os.makedirs(out_dir, exist_ok=True)

available_ids = sorted(df["person_ID"].unique())

for chosen_id in available_ids:
    data = df[df["person_ID"] == chosen_id].copy().reset_index(drop=True)
    depression_label = "Depressed" if data["depression"].iloc[0] == 1 else "Not Depressed"

    # Gaze direction
    gx = mid(data["x_0"], data["x_1"])
    gy = mid(data["y_0"], data["y_1"])
    # Head direction
    hx = mid(data["x_h0"], data["x_h1"])
    hy = mid(data["y_h0"], data["y_h1"])

    # Normalised timestamp colour (0 = light blue, 1 = dark blue)
    n = len(data)
    t_norm = np.linspace(0, 1, n)
    colors = cm.Blues(0.2 + 0.8 * t_norm)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Gaze vs Head Direction — Person {chosen_id} ({depression_label})")

    # Gaze 2D scatter with timestamp gradient
    ax1.scatter(gx, gy, c=colors, alpha=0.6, s=5, edgecolors="none")
    ax1.set_title("Gaze Direction")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.grid(True, alpha=0.3)

    # Head 2D scatter with timestamp gradient
    ax2.scatter(hx, hy, c=colors, alpha=0.6, s=5, edgecolors="none")
    ax2.set_title("Head Direction")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.grid(True, alpha=0.3)

    for ax in (ax1, ax2):
        ax.set_xlim(-global_lim, global_lim)
        ax.set_ylim(-global_lim, global_lim)

    # Horizontal colorbar below both plots showing actual frame numbers
    plt.tight_layout(rect=[0, 0.08, 1, 1]) # leave room at bottom for colorbar

    sm = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(vmin=0, vmax=n))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.15, 0.03, 0.7, 0.025]) # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Frame (early → late)")
    tick_positions = np.linspace(0, n, 6).astype(int)
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(tick_positions)

    out_path = os.path.join(out_dir, f"gaze_scatter_person_{chosen_id}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

print(f"\nDone. {len(available_ids)} plots saved to '{out_dir}/'")