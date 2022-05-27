import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import fnmatch


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
    tmp = []
    for point in ts:
        tmp.append(point[1])
    ts = np.array(tmp)
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


def visualize_with_anomaly_score(data, target_first, predicted_first, target_second, predicted_second, target_third,
                                 predicted_third, target_fourth, predicted_fourth, test_start, anomaly, name=None):
    '''
       Input:
       data: array with the raw data
       test_start: the offset where the test data starts
       predicted: The offset of your prediction.
       anomaly: The offset of the anomaly.
                      If -1 is passed, no anomaly is plottet.
    '''
    anomaly_start = anomaly - 50
    anomaly_end = anomaly + 50

    fig, ax = plt.subplots(9, 1, figsize=(20, 4), sharex=True)
    sns.lineplot(x=np.arange(test_start, len(data)), y=data[test_start:], ax=ax[0], lw=0.5, label="Test")
    sns.lineplot(x=np.arange(0, test_start), y=data[:test_start], ax=ax[0], lw=0.5, label="Train")

    if anomaly_start > 0:
        sns.lineplot(x=np.arange(anomaly_start, anomaly_end),
                     y=data[anomaly_start:anomaly_end], ax=ax[0], label="Actual")

    sns.lineplot(x=np.arange(len(target_first)), y=target_first, ax=ax[1], label="target first")
    sns.lineplot(x=np.arange(len(predicted_first)), y=predicted_first, ax=ax[2], label="predicted first")

    sns.lineplot(x=np.arange(len(target_second)), y=target_second, ax=ax[3], label="target second")
    sns.lineplot(x=np.arange(len(predicted_second)), y=predicted_second, ax=ax[4], label="predicted second")

    sns.lineplot(x=np.arange(len(target_third)), y=target_third, ax=ax[5], label="target third")
    sns.lineplot(x=np.arange(len(predicted_third)), y=predicted_third, ax=ax[6], label="predicted third")

    sns.lineplot(x=np.arange(len(target_fourth)), y=target_fourth, ax=ax[7], label="target fourth")
    sns.lineplot(x=np.arange(len(predicted_fourth)), y=predicted_fourth, ax=ax[8], label="predicted fourth")

    if name is not None:
        ax[0].set_title(name)

    sns.despine()

    #################

    plt.legend()
    plt.show()


def get_anomaly_meta(labels, file):
    locations = pd.read_csv(labels)
    locations.set_index("Name", inplace=True)
    file_name = str(file).split(".")[0].split("_enriched")[0]

    if file_name in locations.index:
        row = locations.loc[file_name]
        test_start = row["TrainSplit"] - 1
        anomaly_index = row["Pos"] - 1
    else:
        raise "Cannot find anomaly_index"

    return anomaly_index, test_start


def enrich_csv_data(labels, anomalie_files):
    locations = pd.read_csv(labels)
    locations.set_index("Name", inplace=True)

    for file in np.sort(fnmatch.filter(os.listdir(anomalie_files), "*.csv")):

        if "Anomaly" in str(file) and "enriched" not in str(file):
            file_name = str(file).split(".")[0]
            if file_name in locations.index:
                row = locations.loc[file_name]
                anomaly_index = row["Pos"]-1
            else:
                raise "Cannot find anomaly_index"

            intermediate_json = {
                "x_axis": {},
                "y_axis": {},
                "is_anomaly": {}
            }

            raw_csv_data = pd.read_csv(anomalie_files + "/" + file, header=None)

            row_counter = 0

            for index, row in raw_csv_data.iterrows():
                intermediate_json["x_axis"][str(row_counter)] = str(row_counter)
                intermediate_json["y_axis"][str(row_counter)] = row.values[0]

                if anomaly_index - 50 <= row_counter <= anomaly_index + 50:
                    intermediate_json["is_anomaly"][str(row_counter)] = 1
                else:
                    intermediate_json["is_anomaly"][str(row_counter)] = 0

                row_counter += 1

            df = pd.DataFrame(intermediate_json)
            new_file_name = file_name + "_enriched.csv"
            df.to_csv("./phase_1/" + new_file_name, index=False)

            print("INFO: Enriched csv files with anomaly meta-data! Created/Updated " + new_file_name)
