from DeepEEG import input_preparation
from DeepEEG.input_preparation import get_filepaths
from DeepEEG.input_preparation import get_labels
from DeepEEG.input_preparation import get_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_list(filepaths):
    list = []
    for i in filepaths.keys():
        list.append(i)
    return (list)


def get_list_with_index(list_data_dict):
    for i, data_dict in enumerate(list_data_dict):
        print(f"Index: {i}, Dictionary: {data_dict}")


def add_column_to_dataframe(dataframe, column_name, values):
    """
    Add a new column to a DataFrame.

    Args:
    - dataframe: The pandas DataFrame to which the 
    column will be added.
    - column_name: Name of the new column.
    - values: Values to be assigned to the new column.

    Returns:
    - Updated DataFrame with the new column.
    """
    dataframe = dataframe.assign(**{column_name: values})
    return dataframe


def load_activity_data(activity, file_prefix, num_files):
    data_list = []
    for i in range(1, num_files + 1):
        file_path = f"Data/{activity}/{file_prefix} {i}cleaned.csv"
        data = np.genfromtxt(file_path, delimiter=',')
        data_list.append(data)
    return np.concatenate(data_list, axis=0)


# Function for sliding window with labels using pandas DataFrames
def sliding_window_with_labels(window_size, signal, labels):
    segments = []
    segment_labels = []
    for i in range(0, len(signal) - window_size + 1,
                   window_size):
        segment = signal.iloc[i:i + window_size]
        # Assuming signal is a DataFrame
        segment_label = labels.iloc[i + window_size - 1]
        # Assuming labels is a DataFrame
        segments.append(segment.values)
        segment_labels.append(max(segment_label))
        # Accessing the label value assuming it's
        # in the first column
    return np.array(segments), np.array(segment_labels)
