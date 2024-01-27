# %%
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from DeepEEG import input_preparation
from DeepEEG.input_preparation import get_filepaths
from DeepEEG.input_preparation import get_labels
from DeepEEG.input_preparation import get_data
import pandas as pd
import numpy as np


# %%
root_folder = "Data"
training_filepaths = get_filepaths(root_folder)
labels = get_labels(root_folder)
training_filepaths, labels


# %%
def get_list(filepaths):
    list = []
    for i in filepaths.keys():
        list.append(i)
    return (list)


# %%
list_data_dict = get_list(training_filepaths)
for i, data_dict in enumerate(list_data_dict):
    print(f"Index: {i}, Dictionary: {data_dict}")


# %%
datasignals_s_1, one_hot_s_1, label_s_1 = get_data(
    list_data_dict[0],
    labels, training_filepaths)
datasignals_s_2, one_hot_s_2, label_s_2 = get_data(
    list_data_dict[3], labels, training_filepaths)
datasignals_s_3, one_hot_s_3, label_s_3 = get_data(
    list_data_dict[1], labels, training_filepaths)
datasignals_s_4, one_hot_s_4, label_s_4 = get_data(
    list_data_dict[2], labels, training_filepaths)
datasignals_r_1, one_hot_r_1, label_r_1 = get_data(
    list_data_dict[8],
    labels, training_filepaths)

datasignals_r_2, one_hot_r_2, label_r_2 = get_data(
    list_data_dict[9], labels, training_filepaths)
datasignals_r_3, one_hot_r_3, label_r_3 = get_data(
    list_data_dict[5], labels, training_filepaths)
datasignals_r_4, one_hot_r_4, label_r_4 = get_data(
    list_data_dict[6], labels, training_filepaths)
datasignals_r_5, one_hot_r_5, label_r_5 = get_data(
    list_data_dict[4], labels, training_filepaths)
datasignals_r_6, one_hot_r_6, label_r_6 = get_data(
    list_data_dict[7], labels, training_filepaths)


datasignals_w_1, one_hot_w_1, label_w_1 = get_data(
    list_data_dict[10], labels, training_filepaths)
datasignals_w_2, one_hot_w_2, label_w_2 = get_data(
    list_data_dict[12], labels, training_filepaths)

datasignals_w_3, one_hot_w_3, label_w_3 = get_data(
    list_data_dict[14], labels, training_filepaths)

datasignals_w_4, one_hot_w_4, label_w_4 = get_data(
    list_data_dict[13], labels, training_filepaths)

datasignals_w_5, one_hot_w_5, label_w_5 = get_data(
    list_data_dict[11], labels, training_filepaths)

datasignals_w_6, one_hot_w_6, label_w_6 = get_data(
    list_data_dict[15], labels, training_filepaths)


datasignals_w_6, one_hot_w_6, label_w_6

# %%
concatenated_data_s = pd.concat([datasignals_s_1,
                                 datasignals_s_2, datasignals_s_3, datasignals_s_4], axis=0,
                                ignore_index=True)

concatenated_data_r = pd.concat([datasignals_r_1,
                                 datasignals_r_2, datasignals_r_3, datasignals_r_4,
                                 datasignals_r_5, datasignals_r_6], axis=0,
                                ignore_index=True)

concatenated_data_w = pd.concat([
    datasignals_w_1, datasignals_w_2, datasignals_w_3,
    datasignals_w_4, datasignals_w_5, datasignals_w_6
], axis=0, ignore_index=True)


# %%
all_Data_connected = pd.concat([
    concatenated_data_s, concatenated_data_r, concatenated_data_w
], axis=0, ignore_index=True)

all_Data_connected.describe(),

# %%

# Assuming you have concatenated your data into concatenated_data DataFrame
# Extracting EEG1, EEG2, Acc_X, Acc_Y, Acc_Z data
eeg1_data = concatenated_data_s['EEG1']
eeg2_data = concatenated_data_s['EEG2']
acc_x_data = concatenated_data_s['Acc_X']
acc_y_data = concatenated_data_s['Acc_Y']
acc_z_data = concatenated_data_s['Acc_Z']

# Create subplots
fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.03)

# Add EEG1 trace to subplot 1
fig.add_trace(go.Scatter(x=concatenated_data_s.index,
              y=eeg1_data, mode='lines', name='EEG1'), row=1, col=1)

# Add EEG2 trace to subplot 2
fig.add_trace(go.Scatter(x=concatenated_data_s.index,
              y=eeg2_data, mode='lines', name='EEG2'), row=2, col=1)

# Add Acc_X trace to subplot 3
fig.add_trace(go.Scatter(x=concatenated_data_s.index,
              y=acc_x_data, mode='lines', name='Acc_X'), row=3, col=1)

# Add Acc_Y trace to subplot 4
fig.add_trace(go.Scatter(x=concatenated_data_s.index,
              y=acc_y_data, mode='lines', name='Acc_Y'), row=4, col=1)

# Add Acc_Z trace to subplot 5
fig.add_trace(go.Scatter(x=concatenated_data_s.index,
              y=acc_z_data, mode='lines', name='Acc_Z'), row=5, col=1)

# Update layout
fig.update_layout(
    height=800,
    title_text="EEG and Accelerometer Signals speaking data",
    xaxis=dict(title='Index'),
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)
fig.show()

# %%

# Assuming you have concatenated your data into concatenated_data DataFrame
# Assuming the data is indexed by the sample number and contains EEG1, EEG2, Acc_X, Acc_Y, Acc_Z data in microvolts (µV)

# Calculate time in seconds based on the sampling rate (220 Hz)
sampling_rate = 220  # Hz
time_seconds = concatenated_data_s.index / sampling_rate

# Extracting EEG1, EEG2, Acc_X, Acc_Y, Acc_Z data
eeg1_data = concatenated_data_s['EEG1']
eeg2_data = concatenated_data_s['EEG2']
acc_x_data = concatenated_data_s['Acc_X']
acc_y_data = concatenated_data_s['Acc_Y']
acc_z_data = concatenated_data_s['Acc_Z']

# Create subplots
fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.03)

# Add EEG1 trace to subplot 1
fig.add_trace(go.Scatter(x=time_seconds, y=eeg1_data,
              mode='lines', name='EEG1'), row=1, col=1)

# Add EEG2 trace to subplot 2
fig.add_trace(go.Scatter(x=time_seconds, y=eeg2_data,
              mode='lines', name='EEG2'), row=2, col=1)

# Add Acc_X trace to subplot 3
fig.add_trace(go.Scatter(x=time_seconds, y=acc_x_data,
              mode='lines', name='Acc_X'), row=3, col=1)

# Add Acc_Y trace to subplot 4
fig.add_trace(go.Scatter(x=time_seconds, y=acc_y_data,
              mode='lines', name='Acc_Y'), row=4, col=1)

# Add Acc_Z trace to subplot 5
fig.add_trace(go.Scatter(x=time_seconds, y=acc_z_data,
              mode='lines', name='Acc_Z'), row=5, col=1)

# Update layout with legends and titles
fig.update_layout(
    height=800,
    title_text="EEG and Accelerometer Signals speaking data",
    xaxis=dict(title='Time (seconds)'),
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)


fig.show()


# %%

# Assuming you have concatenated your data into concatenated_data DataFrame
# Extracting EEG1, EEG2, Acc_X, Acc_Y, Acc_Z data
eeg1_data = all_Data_connected['EEG1']
eeg2_data = all_Data_connected['EEG2']
acc_x_data = all_Data_connected['Acc_X']
acc_y_data = all_Data_connected['Acc_Y']
acc_z_data = all_Data_connected['Acc_Z']

# Create subplots
fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.03)

# Add EEG1 trace to subplot 1
fig.add_trace(go.Scatter(x=all_Data_connected.index, y=eeg1_data,
              mode='lines', name='EEG1'), row=1, col=1)

# Add EEG2 trace to subplot 2
fig.add_trace(go.Scatter(x=all_Data_connected.index, y=eeg2_data,
              mode='lines', name='EEG2'), row=2, col=1)

# Add Acc_X trace to subplot 3
fig.add_trace(go.Scatter(x=all_Data_connected.index, y=acc_x_data,
              mode='lines', name='Acc_X'), row=3, col=1)

# Add Acc_Y trace to subplot 4
fig.add_trace(go.Scatter(x=all_Data_connected.index, y=acc_y_data,
              mode='lines', name='Acc_Y'), row=4, col=1)

# Add Acc_Z trace to subplot 5
fig.add_trace(go.Scatter(x=all_Data_connected.index, y=acc_z_data,
              mode='lines', name='Acc_Z'), row=5, col=1)

# Update layout with legends and titles
fig.update_layout(
    height=800,
    title_text="EEG and Accelerometer Signals all Data",
    xaxis=dict(title='Index'),
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)
fig.show()


# %%

# Assuming you have concatenated
# your data into concatenated_data DataFrame

# Calculate time in seconds based on the sampling rate (220 Hz)
sampling_rate = 220  # Hz
time_seconds = all_Data_connected.index / sampling_rate

# Extracting EEG1, EEG2, Acc_X, Acc_Y, Acc_Z data
eeg1_data = all_Data_connected['EEG1']
eeg2_data = all_Data_connected['EEG2']
acc_x_data = all_Data_connected['Acc_X']
acc_y_data = all_Data_connected['Acc_Y']
acc_z_data = all_Data_connected['Acc_Z']

# Create subplots
fig = make_subplots(rows=5, cols=1,
                    shared_xaxes=True, vertical_spacing=0.03)

# Add EEG1 trace to subplot 1
fig.add_trace(go.Scatter(x=time_seconds,
                         y=eeg1_data, mode='lines', name='EEG1'), row=1, col=1)

# Add EEG2 trace to subplot 2
fig.add_trace(go.Scatter(x=time_seconds,
                         y=eeg2_data, mode='lines', name='EEG2'), row=2, col=1)

# Add Acc_X trace to subplot 3
fig.add_trace(go.Scatter(x=time_seconds,
                         y=acc_x_data, mode='lines', name='Acc_X'), row=3, col=1)

# Add Acc_Y trace to subplot 4
fig.add_trace(go.Scatter(x=time_seconds,
                         y=acc_y_data, mode='lines', name='Acc_Y'), row=4, col=1)

# Add Acc_Z trace to subplot 5
fig.add_trace(go.Scatter(x=time_seconds,
                         y=acc_z_data, mode='lines', name='Acc_Z'), row=5, col=1)

# Update layout with legends and titles
fig.update_layout(
    height=800,
    title_text="EEG and Accelerometer Signals all",
    xaxis=dict(title='Time (seconds)'),
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)


fig.show()

# %% [markdown]
# sliding window approach: divid the data in subsets
# use the index to navigate to make the size of window fixed
# like example
# A = [2,4,6,9,11,13]
# label = [gerade zahl, ungerade zahl]
# labels = []
# you want to make
# window_1 = [2,4,6], 3 numbers, label_1: gerade Zahl
# window_2 = [9,11,13] 3 numbers, label_2: ungerade Zahl
# we use Dictionary
# dict.keys(), dict.values()

# %%


def load_activity_data(activity, file_prefix, num_files):
    data_list = []
    for i in range(1, num_files + 1):
        file_path = f"Data/{activity}/{file_prefix} {i}cleaned.csv"
        data = np.genfromtxt(file_path, delimiter=',')
        data_list.append(data)
    return np.concatenate(data_list, axis=0)


# Load data for each activity
reading_data = load_activity_data("Reading", "r", 6)
speaking_data = load_activity_data("Speaking", "s", 4)
watching_data = load_activity_data("Watching", "w", 6)

# Create labels for each activity

# 0 represents reading
reading_labels = np.full((reading_data.shape[0],), fill_value=0)
# 1 represents speaking
speaking_labels = np.full((speaking_data.shape[0],), fill_value=1)
# 2 represents watching
watching_labels = np.full((watching_data.shape[0],), fill_value=2)

# Concatenate data and labels
X = np.concatenate((reading_data,
                    speaking_data, watching_data), axis=0)
y = np.concatenate((reading_labels,
                    speaking_labels, watching_labels), axis=0)


# %%
label = pd.DataFrame(y, columns=['Label'])
label.shape, all_Data_connected.shape

# %%
data = pd.concat([all_Data_connected, label], axis=1)
data

# %%

# Function for sliding window with labels using pandas DataFrames


def sliding_window_with_labels(window_size, signal, labels):
    segments = []
    segment_labels = []
    for i in range(0, len(signal) - window_size + 1, window_size):
        # Assuming signal is a DataFrame
        segment = signal.iloc[i:i + window_size]
        # Assuming labels is a DataFrame
        segment_label = labels.iloc[i + window_size - 1]
        segments.append(segment.values)
        # Accessing the label value assuming it's in the first column
        segment_labels.append(max(segment_label))
    return np.array(segments), np.array(segment_labels)


# %%
window_size = 100  # Define your window size

# Apply sliding window function with labels
X, y = sliding_window_with_labels(window_size, all_Data_connected, label)


# %%

np.shape(X), np.shape(y)

# %%
X[0, :, 1], y[0]
