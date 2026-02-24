import pandas as pd

def prepare_data(file, metric):
    """
    Takes file path and reads data 
    Returns data filtered for relevant metrics
    """

    data = pd.read_csv(file)
    data_filtered = data[
        (data["stat"]==metric) &
        ~(data["segment_type"]=="all") &
        ~data["AU"].str.endswith("_c")]
    return data_filtered