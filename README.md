

```python
from sklearn import datasets
import numpy as np
import pandas as pd

import plotly.plotly as py
from plotly.graph_objs import *
```


```python
# load the iris dataset
iris = datasets.load_iris()
```


```python
# check out the iris dataset
iris
```




    {'DESCR': 'Iris Plants Database\n====================\n\nNotes\n-----\nData Set Characteristics:\n    :Number of Instances: 150 (50 in each of three classes)\n    :Number of Attributes: 4 numeric, predictive attributes and the class\n    :Attribute Information:\n        - sepal length in cm\n        - sepal width in cm\n        - petal length in cm\n        - petal width in cm\n        - class:\n                - Iris-Setosa\n                - Iris-Versicolour\n                - Iris-Virginica\n    :Summary Statistics:\n\n    ============== ==== ==== ======= ===== ====================\n                    Min  Max   Mean    SD   Class Correlation\n    ============== ==== ==== ======= ===== ====================\n    sepal length:   4.3  7.9   5.84   0.83    0.7826\n    sepal width:    2.0  4.4   3.05   0.43   -0.4194\n    petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)\n    petal width:    0.1  2.5   1.20  0.76     0.9565  (high!)\n    ============== ==== ==== ======= ===== ====================\n\n    :Missing Attribute Values: None\n    :Class Distribution: 33.3% for each of 3 classes.\n    :Creator: R.A. Fisher\n    :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)\n    :Date: July, 1988\n\nThis is a copy of UCI ML iris datasets.\nhttp://archive.ics.uci.edu/ml/datasets/Iris\n\nThe famous Iris database, first used by Sir R.A Fisher\n\nThis is perhaps the best known database to be found in the\npattern recognition literature.  Fisher\'s paper is a classic in the field and\nis referenced frequently to this day.  (See Duda & Hart, for example.)  The\ndata set contains 3 classes of 50 instances each, where each class refers to a\ntype of iris plant.  One class is linearly separable from the other 2; the\nlatter are NOT linearly separable from each other.\n\nReferences\n----------\n   - Fisher,R.A. "The use of multiple measurements in taxonomic problems"\n     Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to\n     Mathematical Statistics" (John Wiley, NY, 1950).\n   - Duda,R.O., & Hart,P.E. (1973) Pattern Classification and Scene Analysis.\n     (Q327.D83) John Wiley & Sons.  ISBN 0-471-22361-1.  See page 218.\n   - Dasarathy, B.V. (1980) "Nosing Around the Neighborhood: A New System\n     Structure and Classification Rule for Recognition in Partially Exposed\n     Environments".  IEEE Transactions on Pattern Analysis and Machine\n     Intelligence, Vol. PAMI-2, No. 1, 67-71.\n   - Gates, G.W. (1972) "The Reduced Nearest Neighbor Rule".  IEEE Transactions\n     on Information Theory, May 1972, 431-433.\n   - See also: 1988 MLC Proceedings, 54-64.  Cheeseman et al"s AUTOCLASS II\n     conceptual clustering system finds 3 classes in the data.\n   - Many, many more ...\n',
     'data': array([[ 5.1,  3.5,  1.4,  0.2],
            [ 4.9,  3. ,  1.4,  0.2],
            [ 4.7,  3.2,  1.3,  0.2],
            [ 4.6,  3.1,  1.5,  0.2],
            [ 5. ,  3.6,  1.4,  0.2],
            [ 5.4,  3.9,  1.7,  0.4],
            [ 4.6,  3.4,  1.4,  0.3],
            [ 5. ,  3.4,  1.5,  0.2],
            [ 4.4,  2.9,  1.4,  0.2],
            [ 4.9,  3.1,  1.5,  0.1],
            [ 5.4,  3.7,  1.5,  0.2],
            [ 4.8,  3.4,  1.6,  0.2],
            [ 4.8,  3. ,  1.4,  0.1],
            [ 4.3,  3. ,  1.1,  0.1],
            [ 5.8,  4. ,  1.2,  0.2],
            [ 5.7,  4.4,  1.5,  0.4],
            [ 5.4,  3.9,  1.3,  0.4],
            [ 5.1,  3.5,  1.4,  0.3],
            [ 5.7,  3.8,  1.7,  0.3],
            [ 5.1,  3.8,  1.5,  0.3],
            [ 5.4,  3.4,  1.7,  0.2],
            [ 5.1,  3.7,  1.5,  0.4],
            [ 4.6,  3.6,  1. ,  0.2],
            [ 5.1,  3.3,  1.7,  0.5],
            [ 4.8,  3.4,  1.9,  0.2],
            [ 5. ,  3. ,  1.6,  0.2],
            [ 5. ,  3.4,  1.6,  0.4],
            [ 5.2,  3.5,  1.5,  0.2],
            [ 5.2,  3.4,  1.4,  0.2],
            [ 4.7,  3.2,  1.6,  0.2],
            [ 4.8,  3.1,  1.6,  0.2],
            [ 5.4,  3.4,  1.5,  0.4],
            [ 5.2,  4.1,  1.5,  0.1],
            [ 5.5,  4.2,  1.4,  0.2],
            [ 4.9,  3.1,  1.5,  0.1],
            [ 5. ,  3.2,  1.2,  0.2],
            [ 5.5,  3.5,  1.3,  0.2],
            [ 4.9,  3.1,  1.5,  0.1],
            [ 4.4,  3. ,  1.3,  0.2],
            [ 5.1,  3.4,  1.5,  0.2],
            [ 5. ,  3.5,  1.3,  0.3],
            [ 4.5,  2.3,  1.3,  0.3],
            [ 4.4,  3.2,  1.3,  0.2],
            [ 5. ,  3.5,  1.6,  0.6],
            [ 5.1,  3.8,  1.9,  0.4],
            [ 4.8,  3. ,  1.4,  0.3],
            [ 5.1,  3.8,  1.6,  0.2],
            [ 4.6,  3.2,  1.4,  0.2],
            [ 5.3,  3.7,  1.5,  0.2],
            [ 5. ,  3.3,  1.4,  0.2],
            [ 7. ,  3.2,  4.7,  1.4],
            [ 6.4,  3.2,  4.5,  1.5],
            [ 6.9,  3.1,  4.9,  1.5],
            [ 5.5,  2.3,  4. ,  1.3],
            [ 6.5,  2.8,  4.6,  1.5],
            [ 5.7,  2.8,  4.5,  1.3],
            [ 6.3,  3.3,  4.7,  1.6],
            [ 4.9,  2.4,  3.3,  1. ],
            [ 6.6,  2.9,  4.6,  1.3],
            [ 5.2,  2.7,  3.9,  1.4],
            [ 5. ,  2. ,  3.5,  1. ],
            [ 5.9,  3. ,  4.2,  1.5],
            [ 6. ,  2.2,  4. ,  1. ],
            [ 6.1,  2.9,  4.7,  1.4],
            [ 5.6,  2.9,  3.6,  1.3],
            [ 6.7,  3.1,  4.4,  1.4],
            [ 5.6,  3. ,  4.5,  1.5],
            [ 5.8,  2.7,  4.1,  1. ],
            [ 6.2,  2.2,  4.5,  1.5],
            [ 5.6,  2.5,  3.9,  1.1],
            [ 5.9,  3.2,  4.8,  1.8],
            [ 6.1,  2.8,  4. ,  1.3],
            [ 6.3,  2.5,  4.9,  1.5],
            [ 6.1,  2.8,  4.7,  1.2],
            [ 6.4,  2.9,  4.3,  1.3],
            [ 6.6,  3. ,  4.4,  1.4],
            [ 6.8,  2.8,  4.8,  1.4],
            [ 6.7,  3. ,  5. ,  1.7],
            [ 6. ,  2.9,  4.5,  1.5],
            [ 5.7,  2.6,  3.5,  1. ],
            [ 5.5,  2.4,  3.8,  1.1],
            [ 5.5,  2.4,  3.7,  1. ],
            [ 5.8,  2.7,  3.9,  1.2],
            [ 6. ,  2.7,  5.1,  1.6],
            [ 5.4,  3. ,  4.5,  1.5],
            [ 6. ,  3.4,  4.5,  1.6],
            [ 6.7,  3.1,  4.7,  1.5],
            [ 6.3,  2.3,  4.4,  1.3],
            [ 5.6,  3. ,  4.1,  1.3],
            [ 5.5,  2.5,  4. ,  1.3],
            [ 5.5,  2.6,  4.4,  1.2],
            [ 6.1,  3. ,  4.6,  1.4],
            [ 5.8,  2.6,  4. ,  1.2],
            [ 5. ,  2.3,  3.3,  1. ],
            [ 5.6,  2.7,  4.2,  1.3],
            [ 5.7,  3. ,  4.2,  1.2],
            [ 5.7,  2.9,  4.2,  1.3],
            [ 6.2,  2.9,  4.3,  1.3],
            [ 5.1,  2.5,  3. ,  1.1],
            [ 5.7,  2.8,  4.1,  1.3],
            [ 6.3,  3.3,  6. ,  2.5],
            [ 5.8,  2.7,  5.1,  1.9],
            [ 7.1,  3. ,  5.9,  2.1],
            [ 6.3,  2.9,  5.6,  1.8],
            [ 6.5,  3. ,  5.8,  2.2],
            [ 7.6,  3. ,  6.6,  2.1],
            [ 4.9,  2.5,  4.5,  1.7],
            [ 7.3,  2.9,  6.3,  1.8],
            [ 6.7,  2.5,  5.8,  1.8],
            [ 7.2,  3.6,  6.1,  2.5],
            [ 6.5,  3.2,  5.1,  2. ],
            [ 6.4,  2.7,  5.3,  1.9],
            [ 6.8,  3. ,  5.5,  2.1],
            [ 5.7,  2.5,  5. ,  2. ],
            [ 5.8,  2.8,  5.1,  2.4],
            [ 6.4,  3.2,  5.3,  2.3],
            [ 6.5,  3. ,  5.5,  1.8],
            [ 7.7,  3.8,  6.7,  2.2],
            [ 7.7,  2.6,  6.9,  2.3],
            [ 6. ,  2.2,  5. ,  1.5],
            [ 6.9,  3.2,  5.7,  2.3],
            [ 5.6,  2.8,  4.9,  2. ],
            [ 7.7,  2.8,  6.7,  2. ],
            [ 6.3,  2.7,  4.9,  1.8],
            [ 6.7,  3.3,  5.7,  2.1],
            [ 7.2,  3.2,  6. ,  1.8],
            [ 6.2,  2.8,  4.8,  1.8],
            [ 6.1,  3. ,  4.9,  1.8],
            [ 6.4,  2.8,  5.6,  2.1],
            [ 7.2,  3. ,  5.8,  1.6],
            [ 7.4,  2.8,  6.1,  1.9],
            [ 7.9,  3.8,  6.4,  2. ],
            [ 6.4,  2.8,  5.6,  2.2],
            [ 6.3,  2.8,  5.1,  1.5],
            [ 6.1,  2.6,  5.6,  1.4],
            [ 7.7,  3. ,  6.1,  2.3],
            [ 6.3,  3.4,  5.6,  2.4],
            [ 6.4,  3.1,  5.5,  1.8],
            [ 6. ,  3. ,  4.8,  1.8],
            [ 6.9,  3.1,  5.4,  2.1],
            [ 6.7,  3.1,  5.6,  2.4],
            [ 6.9,  3.1,  5.1,  2.3],
            [ 5.8,  2.7,  5.1,  1.9],
            [ 6.8,  3.2,  5.9,  2.3],
            [ 6.7,  3.3,  5.7,  2.5],
            [ 6.7,  3. ,  5.2,  2.3],
            [ 6.3,  2.5,  5. ,  1.9],
            [ 6.5,  3. ,  5.2,  2. ],
            [ 6.2,  3.4,  5.4,  2.3],
            [ 5.9,  3. ,  5.1,  1.8]]),
     'feature_names': ['sepal length (cm)',
      'sepal width (cm)',
      'petal length (cm)',
      'petal width (cm)'],
     'target': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
     'target_names': array(['setosa', 'versicolor', 'virginica'],
           dtype='<U10')}




```python
iris.feature_names
```




    ['sepal length (cm)',
     'sepal width (cm)',
     'petal length (cm)',
     'petal width (cm)']




```python
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['target'] = iris.target
```


```python
df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# split up the data and visualize the different classes
setosa = df[df["target"] == 0]
versicolor = df[df["target"] == 1]
virginica = df[df["target"] == 2]
```


```python
# color pallete
colors = ["#E41A1C", "#377EB8", "#4DAF4A" \
          "#984EA3", "#FF7F00", "#FFFF33" \
          "#A65628", "#F781BF", "#999999"]
```


```python
# Do this to pip install from a notebook
# !pip install plotly
```

    Requirement already satisfied: plotly in c:\users\t0nb3\anaconda3\lib\site-packages (2.7.0)
    Requirement already satisfied: pytz in c:\users\t0nb3\anaconda3\lib\site-packages (from plotly) (2017.2)
    Requirement already satisfied: requests in c:\users\t0nb3\anaconda3\lib\site-packages (from plotly) (2.11.1)
    Requirement already satisfied: nbformat>=4.2 in c:\users\t0nb3\anaconda3\lib\site-packages (from plotly) (4.4.0)
    Requirement already satisfied: six in c:\users\t0nb3\anaconda3\lib\site-packages (from plotly) (1.11.0)
    Requirement already satisfied: decorator>=4.0.6 in c:\users\t0nb3\anaconda3\lib\site-packages (from plotly) (4.0.11)
    Requirement already satisfied: ipython_genutils in c:\users\t0nb3\anaconda3\lib\site-packages (from nbformat>=4.2->plotly) (0.2.0)
    Requirement already satisfied: traitlets>=4.1 in c:\users\t0nb3\anaconda3\lib\site-packages (from nbformat>=4.2->plotly) (4.3.2)
    Requirement already satisfied: jsonschema!=2.5.0,>=2.4 in c:\users\t0nb3\anaconda3\lib\site-packages (from nbformat>=4.2->plotly) (2.6.0)
    Requirement already satisfied: jupyter_core in c:\users\t0nb3\anaconda3\lib\site-packages (from nbformat>=4.2->plotly) (4.3.0)
    

    networkx 2.0 has requirement decorator>=4.1.0, but you'll have decorator 4.0.11 which is incompatible.
    


```python
# plot the real labels
trace1 = {
    "x": setosa["sepal length (cm)"],
    "y": setosa["sepal width (cm)"],
    "type": "scatter",
    "mode": "markers",
    "name": "setosa",
    "marker": {
        "color": colors[0],
        "size": 10
    }
}
# or inside the "marker": dict(
#  color = color[0],
#  size = 10
# )

trace2 = {
    "x": versicolor["sepal length (cm)"],
    "y": versicolor["sepal width (cm)"],
    "type": "scatter",
    "mode": "markers",
    "name": "versicolor",
    "marker": {
        "color": colors[1],
        "size": 10
    }
}

trace3 = {
    "x": virginica["sepal length (cm)"],
    "y": virginica["sepal width (cm)"],
    "type": "scatter",
    "mode": "markers",
    "name": "virginica",
    "marker": {
        "color": colors[2],
        "size": 10
    }
}

data = [trace1, trace2 ,trace3]

layout = {
    "hovermode": "closest",
    "margin": {
        "r": 10,
        "t": 25,
        "b": 40,
        "l": 60
    },
    "title": "Iris Dataset - Sepal Length vs Sepal Width",
    "xaxis": {
        "domain": [0, 1],
        "title": "Sepal Length"
    },
    "yaxis": {
        "domain": [0, 1],
        "title": "Sepal Width"
    }
}

fig = Figure(data=data, layout=layout)
py.iplot(fig, filename="iris-scatter")
```

    High five! You successfully sent some data to your account on plotly. View your plot in your browser at https://plot.ly/~totopi/0 or inside your plot.ly account where it is named 'iris-scatter'
    




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~totopi/0.embed" height="525px" width="100%"></iframe>




```python
col = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
X = df[col]
y = df["target"]
```


```python
from sklearn.cluster import KMeans

# declaring our algorithm and its parameters
kmeans = KMeans(n_clusters=6)

# fit data to features
kmeans.fit(X)
```




    KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
        n_clusters=6, n_init=10, n_jobs=1, precompute_distances='auto',
        random_state=None, tol=0.0001, verbose=0)




```python
print(kmeans.labels_[::10])
print(list(y[::10]))
```

    [5 5 5 0 5 2 1 2 1 1 4 4 4 3 4]
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    


```python
results_df = pd.DataFrame(iris.data, columns=iris.feature_names)
results_df["predicted_classes"] = kmeans.labels_
```


```python
results_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
      <th>predicted_classes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
num_of_clusters = results_df['predicted_classes'].nunique()
num_of_clusters
```




    6




```python
data = []
for i in range(num_of_clusters):
    cluster_df = results_df[results_df["predicted_classes"] == i]
    data.append({
        "x": cluster_df["sepal length (cm)"],
        "y": cluster_df["sepal width (cm)"],
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
  "title": "Iris Dataset - Sepal Length vs Sepal Width", 
  "xaxis": {
    "domain": [0, 1], 
    "title": "Sepal Length"
  }, 
  "yaxis": {
    "domain": [0, 1], 
    "title": "Sepal Width"
  }
}

fig = Figure(data=data, layout=layout)
py.iplot(fig, filename='iris-predicted')
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~totopi/2.embed" height="525px" width="100%"></iframe>




```python
data
```




    [{'marker': {'color': '#E41A1C', 'size': 10},
      'mode': 'markers',
      'name': 'class_0',
      'type': 'scatter',
      'x': 52     6.9
      77     6.7
      100    6.3
      102    7.1
      103    6.3
      104    6.5
      105    7.6
      107    7.3
      108    6.7
      109    7.2
      110    6.5
      111    6.4
      112    6.8
      115    6.4
      116    6.5
      117    7.7
      118    7.7
      120    6.9
      122    7.7
      124    6.7
      125    7.2
      128    6.4
      129    7.2
      130    7.4
      131    7.9
      132    6.4
      134    6.1
      135    7.7
      136    6.3
      137    6.4
      139    6.9
      140    6.7
      141    6.9
      143    6.8
      144    6.7
      145    6.7
      147    6.5
      148    6.2
      Name: sepal length (cm), dtype: float64,
      'y': 52     3.1
      77     3.0
      100    3.3
      102    3.0
      103    2.9
      104    3.0
      105    3.0
      107    2.9
      108    2.5
      109    3.6
      110    3.2
      111    2.7
      112    3.0
      115    3.2
      116    3.0
      117    3.8
      118    2.6
      120    3.2
      122    2.8
      124    3.3
      125    3.2
      128    2.8
      129    3.0
      130    2.8
      131    3.8
      132    2.8
      134    2.6
      135    3.0
      136    3.4
      137    3.1
      139    3.1
      140    3.1
      141    3.1
      143    3.2
      144    3.3
      145    3.0
      147    3.0
      148    3.4
      Name: sepal width (cm), dtype: float64},
     {'marker': {'color': '#377EB8', 'size': 10},
      'mode': 'markers',
      'name': 'class_1',
      'type': 'scatter',
      'x': 0     5.1
      1     4.9
      2     4.7
      3     4.6
      4     5.0
      5     5.4
      6     4.6
      7     5.0
      8     4.4
      9     4.9
      10    5.4
      11    4.8
      12    4.8
      13    4.3
      14    5.8
      15    5.7
      16    5.4
      17    5.1
      18    5.7
      19    5.1
      20    5.4
      21    5.1
      22    4.6
      23    5.1
      24    4.8
      25    5.0
      26    5.0
      27    5.2
      28    5.2
      29    4.7
      30    4.8
      31    5.4
      32    5.2
      33    5.5
      34    4.9
      35    5.0
      36    5.5
      37    4.9
      38    4.4
      39    5.1
      40    5.0
      41    4.5
      42    4.4
      43    5.0
      44    5.1
      45    4.8
      46    5.1
      47    4.6
      48    5.3
      49    5.0
      Name: sepal length (cm), dtype: float64,
      'y': 0     3.5
      1     3.0
      2     3.2
      3     3.1
      4     3.6
      5     3.9
      6     3.4
      7     3.4
      8     2.9
      9     3.1
      10    3.7
      11    3.4
      12    3.0
      13    3.0
      14    4.0
      15    4.4
      16    3.9
      17    3.5
      18    3.8
      19    3.8
      20    3.4
      21    3.7
      22    3.6
      23    3.3
      24    3.4
      25    3.0
      26    3.4
      27    3.5
      28    3.4
      29    3.2
      30    3.1
      31    3.4
      32    4.1
      33    4.2
      34    3.1
      35    3.2
      36    3.5
      37    3.1
      38    3.0
      39    3.4
      40    3.5
      41    2.3
      42    3.2
      43    3.5
      44    3.8
      45    3.0
      46    3.8
      47    3.2
      48    3.7
      49    3.3
      Name: sepal width (cm), dtype: float64},
     {'marker': {'color': '#4DAF4A#984EA3', 'size': 10},
      'mode': 'markers',
      'name': 'class_2',
      'type': 'scatter',
      'x': 50     7.0
      51     6.4
      53     5.5
      54     6.5
      55     5.7
      56     6.3
      57     4.9
      58     6.6
      59     5.2
      60     5.0
      61     5.9
      62     6.0
      63     6.1
      64     5.6
      65     6.7
      66     5.6
      67     5.8
      68     6.2
      69     5.6
      70     5.9
      71     6.1
      72     6.3
      73     6.1
      74     6.4
      75     6.6
      76     6.8
      78     6.0
      79     5.7
      80     5.5
      81     5.5
            ... 
      84     5.4
      85     6.0
      86     6.7
      87     6.3
      88     5.6
      89     5.5
      90     5.5
      91     6.1
      92     5.8
      93     5.0
      94     5.6
      95     5.7
      96     5.7
      97     6.2
      98     5.1
      99     5.7
      101    5.8
      106    4.9
      113    5.7
      114    5.8
      119    6.0
      121    5.6
      123    6.3
      126    6.2
      127    6.1
      133    6.3
      138    6.0
      142    5.8
      146    6.3
      149    5.9
      Name: sepal length (cm), Length: 62, dtype: float64,
      'y': 50     3.2
      51     3.2
      53     2.3
      54     2.8
      55     2.8
      56     3.3
      57     2.4
      58     2.9
      59     2.7
      60     2.0
      61     3.0
      62     2.2
      63     2.9
      64     2.9
      65     3.1
      66     3.0
      67     2.7
      68     2.2
      69     2.5
      70     3.2
      71     2.8
      72     2.5
      73     2.8
      74     2.9
      75     3.0
      76     2.8
      78     2.9
      79     2.6
      80     2.4
      81     2.4
            ... 
      84     3.0
      85     3.4
      86     3.1
      87     2.3
      88     3.0
      89     2.5
      90     2.6
      91     3.0
      92     2.6
      93     2.3
      94     2.7
      95     3.0
      96     2.9
      97     2.9
      98     2.5
      99     2.8
      101    2.7
      106    2.5
      113    2.5
      114    2.8
      119    2.2
      121    2.8
      123    2.7
      126    2.8
      127    3.0
      133    2.8
      138    3.0
      142    2.7
      146    2.5
      149    3.0
      Name: sepal width (cm), Length: 62, dtype: float64}]




```python
fig
```




    {'data': [{'marker': {'color': '#E41A1C', 'size': 10},
       'mode': 'markers',
       'name': 'class_0',
       'type': 'scatter',
       'x': 52     6.9
       77     6.7
       100    6.3
       102    7.1
       103    6.3
       104    6.5
       105    7.6
       107    7.3
       108    6.7
       109    7.2
       110    6.5
       111    6.4
       112    6.8
       115    6.4
       116    6.5
       117    7.7
       118    7.7
       120    6.9
       122    7.7
       124    6.7
       125    7.2
       128    6.4
       129    7.2
       130    7.4
       131    7.9
       132    6.4
       134    6.1
       135    7.7
       136    6.3
       137    6.4
       139    6.9
       140    6.7
       141    6.9
       143    6.8
       144    6.7
       145    6.7
       147    6.5
       148    6.2
       Name: sepal length (cm), dtype: float64,
       'y': 52     3.1
       77     3.0
       100    3.3
       102    3.0
       103    2.9
       104    3.0
       105    3.0
       107    2.9
       108    2.5
       109    3.6
       110    3.2
       111    2.7
       112    3.0
       115    3.2
       116    3.0
       117    3.8
       118    2.6
       120    3.2
       122    2.8
       124    3.3
       125    3.2
       128    2.8
       129    3.0
       130    2.8
       131    3.8
       132    2.8
       134    2.6
       135    3.0
       136    3.4
       137    3.1
       139    3.1
       140    3.1
       141    3.1
       143    3.2
       144    3.3
       145    3.0
       147    3.0
       148    3.4
       Name: sepal width (cm), dtype: float64},
      {'marker': {'color': '#377EB8', 'size': 10},
       'mode': 'markers',
       'name': 'class_1',
       'type': 'scatter',
       'x': 0     5.1
       1     4.9
       2     4.7
       3     4.6
       4     5.0
       5     5.4
       6     4.6
       7     5.0
       8     4.4
       9     4.9
       10    5.4
       11    4.8
       12    4.8
       13    4.3
       14    5.8
       15    5.7
       16    5.4
       17    5.1
       18    5.7
       19    5.1
       20    5.4
       21    5.1
       22    4.6
       23    5.1
       24    4.8
       25    5.0
       26    5.0
       27    5.2
       28    5.2
       29    4.7
       30    4.8
       31    5.4
       32    5.2
       33    5.5
       34    4.9
       35    5.0
       36    5.5
       37    4.9
       38    4.4
       39    5.1
       40    5.0
       41    4.5
       42    4.4
       43    5.0
       44    5.1
       45    4.8
       46    5.1
       47    4.6
       48    5.3
       49    5.0
       Name: sepal length (cm), dtype: float64,
       'y': 0     3.5
       1     3.0
       2     3.2
       3     3.1
       4     3.6
       5     3.9
       6     3.4
       7     3.4
       8     2.9
       9     3.1
       10    3.7
       11    3.4
       12    3.0
       13    3.0
       14    4.0
       15    4.4
       16    3.9
       17    3.5
       18    3.8
       19    3.8
       20    3.4
       21    3.7
       22    3.6
       23    3.3
       24    3.4
       25    3.0
       26    3.4
       27    3.5
       28    3.4
       29    3.2
       30    3.1
       31    3.4
       32    4.1
       33    4.2
       34    3.1
       35    3.2
       36    3.5
       37    3.1
       38    3.0
       39    3.4
       40    3.5
       41    2.3
       42    3.2
       43    3.5
       44    3.8
       45    3.0
       46    3.8
       47    3.2
       48    3.7
       49    3.3
       Name: sepal width (cm), dtype: float64},
      {'marker': {'color': '#4DAF4A#984EA3', 'size': 10},
       'mode': 'markers',
       'name': 'class_2',
       'type': 'scatter',
       'x': 50     7.0
       51     6.4
       53     5.5
       54     6.5
       55     5.7
       56     6.3
       57     4.9
       58     6.6
       59     5.2
       60     5.0
       61     5.9
       62     6.0
       63     6.1
       64     5.6
       65     6.7
       66     5.6
       67     5.8
       68     6.2
       69     5.6
       70     5.9
       71     6.1
       72     6.3
       73     6.1
       74     6.4
       75     6.6
       76     6.8
       78     6.0
       79     5.7
       80     5.5
       81     5.5
             ... 
       84     5.4
       85     6.0
       86     6.7
       87     6.3
       88     5.6
       89     5.5
       90     5.5
       91     6.1
       92     5.8
       93     5.0
       94     5.6
       95     5.7
       96     5.7
       97     6.2
       98     5.1
       99     5.7
       101    5.8
       106    4.9
       113    5.7
       114    5.8
       119    6.0
       121    5.6
       123    6.3
       126    6.2
       127    6.1
       133    6.3
       138    6.0
       142    5.8
       146    6.3
       149    5.9
       Name: sepal length (cm), Length: 62, dtype: float64,
       'y': 50     3.2
       51     3.2
       53     2.3
       54     2.8
       55     2.8
       56     3.3
       57     2.4
       58     2.9
       59     2.7
       60     2.0
       61     3.0
       62     2.2
       63     2.9
       64     2.9
       65     3.1
       66     3.0
       67     2.7
       68     2.2
       69     2.5
       70     3.2
       71     2.8
       72     2.5
       73     2.8
       74     2.9
       75     3.0
       76     2.8
       78     2.9
       79     2.6
       80     2.4
       81     2.4
             ... 
       84     3.0
       85     3.4
       86     3.1
       87     2.3
       88     3.0
       89     2.5
       90     2.6
       91     3.0
       92     2.6
       93     2.3
       94     2.7
       95     3.0
       96     2.9
       97     2.9
       98     2.5
       99     2.8
       101    2.7
       106    2.5
       113    2.5
       114    2.8
       119    2.2
       121    2.8
       123    2.7
       126    2.8
       127    3.0
       133    2.8
       138    3.0
       142    2.7
       146    2.5
       149    3.0
       Name: sepal width (cm), Length: 62, dtype: float64}],
     'layout': {'hovermode': 'closest',
      'margin': {'b': 40, 'l': 60, 'r': 10, 't': 25},
      'title': 'Iris Dataset - Sepal Length vs Sepal Width',
      'xaxis': {'domain': [0, 1], 'title': 'Sepal Length'},
      'yaxis': {'domain': [0, 1], 'title': 'Sepal Width'}}}


