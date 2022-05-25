import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import plotly.graph_objects as go
import os
import fnmatch
import zipfile
from skmultiflow.anomaly_detection import HalfSpaceTrees
from skmultiflow.data import FileStream


def read_series(path, file, locations=None):
    raw_data = pd.read_csv(path + "/" + file, header=None)
    raw_data = np.array(raw_data).flatten()

    i = 0
    data = []
    for entry in raw_data:
        data.append([float(i), float(entry)])
        i += 1

    data = np.array(data)

    # Extract file name
    file_name = file.split('.')[0]
    splits = file_name.split('_')
    test_start = np.array(splits[-1])

    # load the anomalies
    if locations is None:
        locations = pd.read_csv("phase_1/labels.csv")
        locations.set_index("Name", inplace=True)

    # Extract anomaly location
    anomaly = [-1, -1]
    if file_name in locations.index:
        row = locations.loc[file_name]
        anomaly = row["Pos"]

    return (file_name, int(test_start), data, anomaly)


def find_dominant_window_sizes(ts, n_size=1):
    fourier = np.absolute(np.fft.fft(ts))
    freq = np.fft.fftfreq(ts.shape[0], 1)
    coefs = []
    window_sizes = []
    for coef, freq in zip(fourier, freq):
        if coef > 0 and freq > 0 and freq < 0.2:
            window_size = 1.0 / freq
            # avoid too large windows
            if (window_size < 500) and (window_size > 10):
                coefs.append(coef)
                window_sizes.append(window_size)
    coefs = np.array(coefs)
    window_sizes = np.asarray(window_sizes, dtype=np.int64)
    idx = np.argsort(coefs)[::-1]

    unique_window_sizes = set()
    for window_size in window_sizes[idx]:
        if len(unique_window_sizes) == n_size:
            return np.array([w for w in unique_window_sizes])
        unique_window_sizes.add(window_size)
    return np.array(list(unique_window_sizes))


def sliding_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def visualize_with_anomaly_score(
        data, score, test_start,
        predicted, anomaly, name=None
):
    '''Input:
       data: array with the raw data
       test_start: the offset where the test data starts
       predicted: The offset of your prediction.
       anomaly: The offset of the anomaly.
                      If -1 is passed, no anomaly is plottet.
    '''

    anomaly_start = anomaly - 50
    anomaly_end = anomaly + 50
    predicted_start = predicted - 50
    predicted_end = predicted + 50

    fig, ax = plt.subplots(2, 1, figsize=(20, 4), sharex=True)
    sns.lineplot(x=np.arange(test_start, len(data)), y=data[test_start:], ax=ax[0], lw=0.5, label="Test")
    sns.lineplot(x=np.arange(0, test_start), y=data[:test_start], ax=ax[0], lw=0.5, label="Train")

    if (anomaly_start > 0):
        sns.lineplot(x=np.arange(anomaly_start, anomaly_end),
                     y=data[anomaly_start:anomaly_end], ax=ax[0], label="Actual")

    sns.lineplot(x=np.arange(len(score)), y=score, ax=ax[1], label="Anomaly Scores")
    sns.lineplot(x=np.arange(predicted_start, predicted_end),
                 y=data[predicted_start:predicted_end], ax=ax[0], label="Predicted")

    if name is not None:
        ax[0].set_title(name)

    sns.despine()

    #################

    plt.legend()
    plt.show()


locations = pd.read_csv("../data_sets/phase_1/labels.csv")
locations.set_index("Name", inplace=True)
anomalie_files = "../data_sets/phase_1"
anomalies = 0
anomalies_correct = 0

for file in np.sort(fnmatch.filter(os.listdir(anomalie_files), "*.csv")):
    if "Anomaly" in str(file):
        name, test_start, data, anomaly = read_series(anomalie_files, file, locations)
        print("name: {0}, test_start: {1}, anomaly: {2}".format(name, test_start, anomaly))

        detected_anomalies = 0
        half_space_trees = HalfSpaceTrees()
        true_positives = 0
        n_samples = 0

        for point in data:
            X = np.array([point])
            if int(point[0]) == anomaly:
                y = [1]
            else:
                y = [0]

            y_pred = half_space_trees.predict(X)
            if int(point[0]) == anomaly:
                true_positives += 1
                if y_pred[0] == 1:
                    detected_anomalies += 1
            n_samples += 1
            half_space_trees.partial_fit(X, y)

            #print("y_pred: {0}".format(y_pred))

        print('{} samples analyzed.'.format(n_samples))
        print('Half-Space Trees correctly detected {} out of {} anomalies'.format(detected_anomalies, true_positives))

        anomalies += 1
        if detected_anomalies == 1:
            anomalies_correct += 1

print("anomalies - anomalies_correct / anomalies".format((anomalies - anomalies_correct)/anomalies))