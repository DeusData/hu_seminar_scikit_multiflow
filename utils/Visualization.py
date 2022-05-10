import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import plotly.graph_objects as go
import os
import fnmatch
import zipfile
import ipywidgets as widgets
from ipywidgets import interact, interact_manual


def read_series(path, file, locations=None):   
    print (path + "/"+ file)
    data = pd.read_csv(path + "/"+ file, header=None)
    data = np.array(data).flatten()
    
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

    return file_name, int(test_start), data, anomaly


anomalie_files = "phase_1"


@interact
def show(file=np.sort(fnmatch.filter(os.listdir(anomalie_files), "*.csv"))):
    name, test_start, data, anomaly = read_series(anomalie_files, file)
        
    # Create figure
    layout = dict(xaxis = dict(showgrid=False, ticks='inside'),
                  yaxis = dict(showgrid=False, ticks='inside'),
                  font=dict(size=12),
                )    
    fig = go.Figure(layout=layout)

    # Train
    fig.add_trace(
        go.Scatter(x=list(range(test_start)), y=data[:test_start],
                   line=dict(width=1, color='green')))
    
    # Test
    fig.add_trace(
        go.Scatter(x=list(range(test_start, len(data))), y=data[test_start:],
                   line=dict(width=1, color='blue')))
    
    # Anomaly
    if anomaly>0:
        fig.add_trace(
            go.Scatter(x=list(range(anomaly-50, anomaly+50)), 
                       y=data[anomaly-50:anomaly+50], 
                       line=dict(width=1, color='red')))

    # Set title
    fig.update_layout(
        title_text="Time Series with Range Slider and Selectors",
        autosize=True,
        width=900,
        height=400,
    )

    # Add range slider
    fig.update_layout(
        xaxis=dict(            
            rangeslider=dict(
                visible=True
            ),
            type="linear"
        )
    )

    # fig.update_layout(template="none")
    fig.show()
