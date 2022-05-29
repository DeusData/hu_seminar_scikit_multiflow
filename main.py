import numpy as np
import os
import json
import fnmatch
import pandas as pd
import re
import utils.baseline as bl
from skmultiflow.trees import StackedSingleTargetHoeffdingTreeRegressor
from skmultiflow.lazy import KNNRegressor
from skmultiflow.trees import iSOUPTreeRegressor
from skmultiflow.trees import HoeffdingTreeRegressor
from skmultiflow.trees import HoeffdingAdaptiveTreeRegressor
from skmultiflow.meta import AdaptiveRandomForestRegressor


def camel_to_snake(lib_name):
    lib_name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', lib_name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', lib_name).lower()


if __name__ == "__main__":
    ## Retrieve initial config
    ## Not ready yet for half-space-trees
    with open("./config.json") as f:
        config = json.load(f)

    locations = pd.read_csv("./phase_1/labels.csv")
    locations.set_index("Name", inplace=True)
    anomalie_files = "./phase_1"

    raw_cumulated_final_csv = {
        "ID": {},
        "PREDICTED": {},
        "LIBRARY": {}
    }

    cumulated_libary_csv_index = 0

    for library in config["libraries"]:
        if not library["to_analyze"]:
            continue

        print("INFO: Analyzing data with library " + library["name"])
        total_score = 0
        predictions = []
        ids = []
        i = 0

        raw_final_csv = {
            "ID": {},
            "PREDICTED": {}
        }

        library_csv_index = 0

        for file in np.sort(fnmatch.filter(os.listdir(anomalie_files), "*.csv")):
            if "Anomaly" in str(file) and "enriched" not in str(file):
                file_name = file.split('.')[0]
                name, test_start, data, anomaly = bl.read_series(anomalie_files, file, locations)

                periods = bl.find_dominant_window_sizes(data[:test_start])
                window_size = np.int32(periods[0] / 4)  # try different factors: 2,3,4,... here!
                X = bl.sliding_window(data, window_size)

                if library["name"] == "StackedSingleTargetHoeffdingTreeRegressor":
                    reg = StackedSingleTargetHoeffdingTreeRegressor()
                elif library["name"] == "KNNRegressor":
                    reg = KNNRegressor(  # try different regressors
                        max_window_size=test_start  # this is where learning stops
                        # try different parameters
                    )
                elif library["name"] == "iSOUPTreeRegressor":
                    reg = iSOUPTreeRegressor()
                elif library["name"] == "HoeffdingTreeRegressor":
                    reg = HoeffdingTreeRegressor()
                elif library["name"] == "HoeffdingAdaptiveTreeRegressor":
                    reg = HoeffdingAdaptiveTreeRegressor()
                elif library["name"] == "AdaptiveRandomForestRegressor":
                    reg = AdaptiveRandomForestRegressor(  # try different regressors
                        max_features=test_start  # this is where learning stops
                        # try different parameters
                    )
                else:
                    raise Exception("Undefined library name, please look up configuration!")

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

                if library["to_visualize"]:
                    # Visualize the predicted anomaly
                    bl.visualize_with_anomaly_score(
                        data, score, test_start, predicted, anomaly, name)

                if anomaly > -1:
                    total_score += abs(anomaly - predicted) / anomaly
                    i = i + 1

                raw_final_csv["ID"][str(library_csv_index)] = file_name
                raw_final_csv["PREDICTED"][str(library_csv_index)] = predicted
                library_csv_index += 1

                raw_cumulated_final_csv["ID"][str(cumulated_libary_csv_index)] = file_name
                raw_cumulated_final_csv["PREDICTED"][str(cumulated_libary_csv_index)] = predicted
                raw_cumulated_final_csv["LIBRARY"][str(cumulated_libary_csv_index)] = library["name"]
                cumulated_libary_csv_index += 1

                print(
                    "INFO: Current Score: {}, anomaly: {}, predicted: {}".format((total_score / i) * 100, anomaly, predicted))

        df = pd.DataFrame(raw_final_csv)
        new_file_name = camel_to_snake(library["name"]) + "_results.csv"
        df.to_csv("./results/" + new_file_name, index=False)
        print("INFO: Wrote results to" + "./results/" + new_file_name)

    df = pd.DataFrame(raw_cumulated_final_csv)
    new_file_name = "_results.csv"
    df.to_csv("./results/cumulated_final_results.csv", index=False)
    print("INFO: Wrote results to" + "./results/cumulated_final_results.csv")
