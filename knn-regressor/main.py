import numpy as np
import os
import fnmatch
import pandas as pd
import utils.baseline as bl
from skmultiflow.lazy import KNNRegressor

if __name__ == "__main__":
    ## TODO: Build new project
    print(__file__)
    locations = pd.read_csv("./phase_1/labels.csv")
    locations.set_index("Name", inplace=True)
    anomalie_files = "./phase_1"

    total_score = 0
    predictions = []
    ids = []
    i = 0

    for file in np.sort(fnmatch.filter(os.listdir(anomalie_files), "*.csv")):
        if "Anomaly" in str(file) and "enriched" not in str(file):
            file_name = file.split('.')[0]
            name, test_start, data, anomaly = bl.read_series(anomalie_files, file, locations)

            periods = bl.find_dominant_window_sizes(data[:test_start])
            window_size = np.int32(periods[0] / 4)  # try different factors: 2,3,4,... here!
            X = bl.sliding_window(data, window_size)

            reg = KNNRegressor(  # try different regressors
                max_window_size=test_start  # this is where learning stops
                # try different parameters
            )

            # training
            for y in X[:test_start]:
                # use the window as input
                xs = [y]

                # try to predict the last value of the window
                reg.partial_fit(xs, y[-1:])
                # you can also try to predict the mean/max/min of the window
                # but do not forget to change it below, too

            # testing
            score = np.zeros(len(X))
            for i, y in enumerate(X[test_start:]):
                # use the window as input
                xs = [y]

                # try to predict the last value of the window
                score[test_start + i] = abs(reg.predict(xs) - y[-1:])

            predicted = test_start + np.argmax(score[test_start:])
            predictions.append(predicted)
            ids.append(file_name)

            score[:test_start] = np.NaN

            # Visualize the predicted anomaly
            bl.visualize_with_anomaly_score(
                data, score, test_start, predicted, anomaly, name)

            if (anomaly > -1):
                total_score += abs(anomaly - predicted) / (anomaly)
                i = i + 1

            print("Current Score: ", (total_score / i) * 100)

    print("\tTotal score:", (total_score / 30) * 100)