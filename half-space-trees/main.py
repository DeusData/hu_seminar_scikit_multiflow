import numpy as np
import os
import fnmatch
from skmultiflow.anomaly_detection import HalfSpaceTrees
from skmultiflow.data import FileStream
import utils.baseline as bl

if __name__ == "__main__":
    # have you labels.csv and data csvs stored in ./phase_1/ folder inside the main folder
    labels_location = "./half-space-trees/phase_1/labels.csv"
    anomaly_location = "./half-space-trees/phase_1"
    # bl.enrich_csv_data(labels=labels_location, anomalie_files=anomaly_location) ## do when you run main for the very first time to create enriched csvs

    count = 0

    for file in np.sort(fnmatch.filter(os.listdir(anomaly_location), "*.csv")):
        if count > 0:  # to not get spammed with graphs. Delete for final version, increase if you want to scan more datasets
            break

        if "Anomaly" in str(file) and "enriched" in str(file):
            print("INFO: Predicting anomaly for file {}".format(file))
            anomaly_center, train_until_index = bl.get_anomaly_meta(labels=labels_location, file=file)

            count += 1  # obligatory count

            stream = FileStream(anomaly_location + "/" + file)
            data_list = []
            target_first = []
            predicted_first = []
            target_second = []
            predicted_second = []
            target_third = []
            predicted_third = []
            target_fourth = []
            predicted_fourth = []

            ## FIRST RUN WITH PARTIAL FITTING AND #1 HalfSpaceTree
            first_half_space_trees = HalfSpaceTrees()   # default window_size = 250
            stream.restart()
            print("INFO: First Run!")
            while stream.has_more_samples():
                X, y = stream.next_sample()
                data_list.append(X[0][1])  # Collecting data only once for viz
                target_first.append(y[0])
                y_pred = first_half_space_trees.predict(X)
                predicted_first.append(y_pred[0])
                first_half_space_trees.partial_fit(X, y)

            ## SECOND RUN WITHOUT PARTIAL FITTING AND #1 HalfSpaceTree
            stream.restart()
            print("INFO: Second Run!")
            while stream.has_more_samples():
                X, y = stream.next_sample()
                target_second.append(y[0])
                y_pred = first_half_space_trees.predict(X)
                predicted_second.append(y_pred[0])

            ## FIRST RUN WITH FITTING AND #2 HalfSpaceTree
            second_half_space_trees = HalfSpaceTrees()   # default window_size = 250
            stream.restart()
            print("INFO: Third Run!")
            while stream.has_more_samples():
                X, y = stream.next_sample()
                target_third.append(y[0])
                y_pred = second_half_space_trees.predict(X)
                predicted_third.append(y_pred[0])
                second_half_space_trees.fit(X, y)

            ## SECOND RUN WITHOUT FITTING AND #2 HalfSpaceTree
            stream.restart()
            print("INFO: Fourth Run!")
            while stream.has_more_samples():
                X, y = stream.next_sample()
                target_fourth.append(y[0])
                y_pred = second_half_space_trees.predict(X)
                predicted_fourth.append(y_pred[0])

            ## VISUALIZATION OF RESULTS
            bl.visualize_with_anomaly_score(data=np.array(data_list),
                                            target_first=np.array(target_first),
                                            predicted_first=np.array(predicted_first),
                                            target_second=np.array(target_second),
                                            predicted_second=np.array(predicted_second),
                                            target_third=np.array(target_third),
                                            predicted_third=np.array(predicted_third),
                                            target_fourth=np.array(target_fourth),
                                            predicted_fourth=np.array(predicted_fourth),
                                            test_start=train_until_index,
                                            anomaly=anomaly_center,
                                            name=str(file))
