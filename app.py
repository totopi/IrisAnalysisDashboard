import dash
import dash_core_components as dcc
import dash_html_components as html
from dash import Dash
from dash.dependencies import Input, Output

from sklearn import datasets
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

import plotly.plotly as py
from plotly.graph_objs import *

app = Dash(__name__)
server = app.server

# do all the machine learning
iris = datasets.load_iris()
colors = ["#E41A1C", "#377EB8", "#4DAF4A", \
          "#984EA3", "#FF7F00", "#FFFF33", \
          "#A65628", "#F781BF", "#999999"]
number_of_clusters = range(10)


df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
col = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
X = df[col]
y = df["target"]

df_list = []
# declaring our algorithm and its parameters
for n in number_of_clusters:
    kmeans = KMeans(n_clusters=n + 1)

    # fit data to features
    kmeans.fit(X)

    results_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    results_df["predicted_classes"] = kmeans.labels_

    num_of_clusters = results_df['predicted_classes'].nunique()
    df_list.append({
        'results_df': results_df,
        'num_of_clusters': num_of_clusters
    })

dropdown_val = iris.feature_names

app.layout = html.Div(className="container", children=[
                html.Div(className="jumbotron text-center", children=[
                    html.H1("Iris Analysis"),
                    html.P("Select the X and Y to visualize the data using chosen number of clusters")
                    ]),
                dcc.Dropdown(className="col-md-4", id="dropdown_x",
                    options=[
                        {'label': val, 'value': val} for val in dropdown_val
                    ],
                    value=dropdown_val[0] #default value for first launch
                ),
                dcc.Dropdown(className="col-md-4", id="dropdown_y",
                    options=[
                        {'label': val, 'value': val} for val in dropdown_val
                    ],
                    value=dropdown_val[1]
                ),
                dcc.Dropdown(className="col-md-4", id="dropdown_k",
                    options=[
                        {'label': str(val+1), 'value': val} for val in number_of_clusters
                    ],
                    value=number_of_clusters[3]
                ),
                html.Div(style={"padding": "20px"}, children=[
                    dcc.Graph(id="cluster")
                ])
    ])

# import external css
app.css.append_css({"external_url": "https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css"})

# import external javascript
app.scripts.append_script({"external_url": "https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js"})

# insert callbacks
@app.callback(Output("cluster", "figure"), [Input('dropdown_x', 'value'),Input('dropdown_y', 'value'),Input('dropdown_k', 'value')])
def update_graph(x, y, k):
    data = []
    num_of_clusters = df_list[k]['num_of_clusters']
    for i in range(num_of_clusters):
        results_df = df_list[k]['results_df']
        cluster_df = results_df[results_df["predicted_classes"] == i]
        data.append({
            "x": cluster_df[x],
            "y": cluster_df[y],
            "type": "scatter",
            "mode": "markers",
            "name": f"class_{i}",
            "marker": dict(
                color = colors[i],
                size = 10
            )
        })
        
    layout = {
        "hovermode": "closest", 
        "margin": {
            "r": 10, 
            "t": 25, 
            "b": 40, 
            "l": 60
        }, 
        "title": f"Iris Dataset - {x} vs {y}", 
        "xaxis": {
            "domain": [0, 1], 
            "title": x
        }, 
        "yaxis": {
            "domain": [0, 1], 
            "title": y
        }
    }

    fig = Figure(data=data, layout=layout)
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)