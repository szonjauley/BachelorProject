import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = pd.read_csv("/Users/szonja/Documents/ITU/Thesis/all_processed_data/all_all/combined_AUs_labeled_all_all.csv") #change this

#clean 
data = data[(data["success"]==1) & (data["confidence"] > 0.7)]
data = data.drop(columns=["speaker", "frame", "timestamp", "confidence", "success", 
                          "AU04_c", "AU12_c", "AU15_c", "AU23_c", "AU28_c", "AU45_c"])
data.columns = [c[:4] if c.startswith("AU") else c for c in data.columns]
clean_data = data.set_index("person_ID")
print(clean_data.head())

#aggregate
grouped = clean_data.groupby(level="person_ID")
aggregated = grouped.agg(["median", "mean", "std"])
long = (
    aggregated
    .stack([0, 1], future_stack=True)
    .rename("value")
    .rename_axis(["person_ID", "AU", "metric"])
    .reset_index())

# Merge depression back in
depression_map = clean_data[["depression"]].drop_duplicates()
long = long.merge(depression_map, on="person_ID")
print(long.head())