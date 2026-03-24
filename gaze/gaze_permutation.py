import pandas as pd

def prepare_data(file):
    """
    Takes file path and reads data 
    Returns data filtered listening/speaking segment types
    """
    data = pd.read_csv(file)

    data_filtered = data[
        (data["segment_type"] != "all")
    ].copy()

    return data_filtered