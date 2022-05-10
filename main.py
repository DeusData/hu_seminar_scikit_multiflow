from skmultiflow.data import AnomalySineGenerator
from skmultiflow.anomaly_detection import HalfSpaceTrees
import pandas as pd
import json


df = pd.read_csv (r'./data_sets/phase_1_2/labels.csv')
df.to_json(r'./labels.json')

with open('./labels.json') as f:
    labels = json.loads(f.read())

print(labels)

# Setup a data stream
stream = AnomalySineGenerator(random_state=1, n_samples=1000, n_anomalies=250)

# Setup Half-Space Trees estimator
half_space_trees = HalfSpaceTrees(random_state=1)

# Setup variables to control loop and track performance
max_samples = 1000
n_samples = 0
true_positives = 0
detected_anomalies = 0

# Train the estimator(s) with the samples provided by the data stream
while n_samples < max_samples and stream.has_more_samples():
    X, y = stream.next_sample()
    y_pred = half_space_trees.predict(X)
    if y[0] == 1:
        true_positives += 1
        if y_pred[0] == 1:
            detected_anomalies += 1
    half_space_trees.partial_fit(X, y)
    n_samples += 1

print('{} samples analyzed.'.format(n_samples))
print('Half-Space Trees correctly detected {} out of {} anomalies'.format(detected_anomalies, true_positives))
