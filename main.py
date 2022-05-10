from skmultiflow.data import AnomalySineGenerator
from skmultiflow.anomaly_detection import HalfSpaceTrees
import pandas as pd
import json


df = pd.read_csv (r'./data_sets/phase_1_2/labels.csv')
df.to_json(r'./labels.json')

with open('./labels.json') as f:
    labels = json.loads(f.read())

print(labels)
