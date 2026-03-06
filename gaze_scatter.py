import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
CSV_PATH = "cleaned_gaze_labeled_0_7.csv"
df = pd.read_csv(CSV_PATH)

# Compute global limits across ALL person IDs
all_x = pd.concat([df["x_0"], df["x_1"], df["x_h0"], df["x_h1"]])
all_y = pd.concat([df["y_0"], df["y_1"], df["y_h0"], df["y_h1"]])
global_lim = max(abs(all_x.min()), abs(all_x.max()), abs(all_y.min()), abs(all_y.max()))

# Choose person ID from the CSV file
available_ids = sorted(df["person_ID"].unique())
print("Available person IDs:", available_ids)
while True:
    try:
        chosen_id = int(input(f"Enter person ID: "))
        if chosen_id in available_ids:
            break
        print(f"{chosen_id} not found. Choose from {available_ids}")
    except ValueError:
        print("Please enter a valid integer.")

data = df[df["person_ID"] == chosen_id].copy()
depression_label = "Depressed" if data["depression"].iloc[0] == 1 else "Not Depressed"
print(f"\nPlotting {len(data):,} rows for person {chosen_id} ({depression_label}) …")

# Man of two landmarks
def mid(a, b):
    return (a + b) / 2

# Gaze direction: average of left-eye (x_0,y_0,z_0) and right-eye (x_1,y_1,z_1)
gaze_x = mid(data["x_0"], data["x_1"])
gaze_y = mid(data["y_0"], data["y_1"])
gaze_z = mid(data["z_0"], data["z_1"])

# Head direction: average of left (x_h0,y_h0,z_h0) and right (x_h1,y_h1,z_h1)
head_x = mid(data["x_h0"], data["x_h1"])
head_y = mid(data["y_h0"], data["y_h1"])
head_z = mid(data["z_h0"], data["z_h1"])

hx, hy, hz = head_x, head_y, head_z
gx, gy, gz = gaze_x, gaze_y, gaze_z

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"Gaze vs Head Direction for Person {chosen_id} ({depression_label})")

# Gaze 2D
ax1.scatter(gx, gy, c="red", alpha=0.5, s=5, edgecolors="none")
ax1.set_title("Gaze Direction")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.grid(True, alpha=0.3)

# Head 2D scatter
ax2.scatter(hx, hy, c="blue", alpha=0.5, s=5, edgecolors="none")
ax2.set_title("Head Direction")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.grid(True, alpha=0.3)

for ax in (ax1, ax2):
    ax.set_xlim(-global_lim, global_lim)
    ax.set_ylim(-global_lim, global_lim)

plt.tight_layout()

plt.savefig(f"gaze_scatter_person_{chosen_id}.png", dpi=150, bbox_inches="tight")
print(f"Plot saved → gaze_scatter_person_{chosen_id}.png")
plt.show()