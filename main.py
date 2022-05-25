import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

import plotly.graph_objects as go
import os
import fnmatch
import zipfile

import ipywidgets as widgets
from ipywidgets import interact, interact_manual

# %matplotlib inline  plt.show() or
# %config InlineBackend.figure_formats = {'png', 'retina'} plt.savefig( 'myfig.png' )


def read_series(path, file, locations=None):
    print(path + "/" + file)
    data = pd.read_csv(path + "/" + file, header=None)
    data = np.array(data).flatten()

    # Extract file name
    file_name = file.split('.')[0]
    splits = file_name.split('_')
    test_start = np.array(splits[-1])

    file_name = None

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
