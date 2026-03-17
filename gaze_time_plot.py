import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

file = pd.read_csv("combined_gaze_deltas_all.csv")
file["time_idx"] = file.groupby("person_ID").cumcount() # count instead of timeframes

color_map = {"Listening": "blue", "Speaking": "red"}
persons = file["person_ID"].unique()

for person in persons:
    person_file = file[file["person_ID"] == person].reset_index(drop=True)

    boundary = person_file["speaker"] != person_file["speaker"].shift()
    person_file["plot_segment"] = boundary.cumsum()

    fig, ax = plt.subplots(figsize=(14, 4))

    for _, segment_df in person_file.groupby("plot_segment"):
        speaker = segment_df["speaker"].iloc[0]
        color = color_map.get(speaker, "gray") # use gray in case no segments are found 
        ax.scatter(segment_df["time_idx"], segment_df["delta_deg"], color=color, s=0.5)

    handles = [
        mpatches.Patch(color="blue", label="Listening"),
        mpatches.Patch(color="red", label="Speaking"),
    ]

    ax.legend(handles=handles)
    ax.set_title(f"Gaze Delta Over Time — Participant {person}")
    ax.set_xlabel("Frame (time proxy)")
    ax.set_ylabel("Delta (degrees)")

    plt.tight_layout()
    plt.savefig(f"gaze_delta_{person}.png", dpi=150)
    plt.close()
    print(f"Saved plot for {person}")