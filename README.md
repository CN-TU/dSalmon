# dSalmon
dSalmon (Data Stream Analysis Algorithms for the Impatient) is a framework for analyzing data streams. Implementation of the core algorithms is done in C++, focusing on superior processing speed and allowing even vast amounts of data to be processed. Python bindings are provided to allow seamless integration in data science development.

Installation
------------
dSalmon can be installed using `pip` by running
```pip install github+https://github.com/CN-TU/dSalmon```

Outlier Detectors
-----------------
dSalmon provides several algorithms for detecting outliers in data streams. Usage is easiest using the Python interface, which provides an interface similar to the algorithms from scikit-learn. The following example performs k-nearest neighbor outlier detection with a window size of 100 samples:
```python
from dSalmon import outlier
import pandas
X = pandas.read_csv('my_dataset.csv')
detector = outlier.SWKNN(window=100,k=5)
outlier_scores = detector.fit_predict(X)
print ('Outlier scores: ', outlier_scores)
```
Individual rows of the passed data are processed sequentially. Hence, while being substantially faster, the above code provides similar results as the following example:
```python
from dSalmon import outlier
import pandas
X = pandas.read_csv('my_dataset.csv')
detector = outlier.SWKNN(window=100,k=5)
outlier_scores = [ detector.fit_predict(X.iloc[i,:]) for i in range(len(X)) ]
print ('Outlier scores: ', outlier_scores)
```

M-Tree usage
------------
dSalmon uses an M-Tree for several of its algorithms. An M-Tree is a spatial indexing data structure for metric spaces, allowing fast nearest-neighbor and range queries. The benefit of an M-Tree compared to, e.g., a BallTree is that insertion, updating and removal of points is fast after having built the tree.
For the development of custom algorithms, an M-Tree interface is provided for Python.
A point within a tree can be accessed either via `tree[k]` using the
point's key `k`, or via `tree.ix[i]` using the point's index `i`.
Keys can be arbitrary integers and are returned by the `insert`, `knn` and
`neighbors` functions. Indices are integers in the range 0...`len(tree)`, sorted
according to the points' keys in ascending order.
The following example finds the nearest neighbors of the first 5 points in a data set.
```python
from dSalmon.trees import MTree
tree = MTree()
X = pandas.read_csv('my_dataset.csv')
tree.insert(X)
print ('Nearest neighbors:', tree.knn(X[:5,:]))
```

Knn queries can be performed using the `knn()` function and range queries can be performed using the `neighbors()` function.

Extending dSalmon
-----------------
dSalmon uses `swig` for generating wrapper code for the C++ core algorithms and instantiates single and double precision floating point algorithms of each algorithm.
The `cpp` folder contains the C++ code, which might be used directly by C++ projects. The `python` folder contains the Python interface invoking the `swig` wrappers and the `swig` folder contains the C++ interface.
When adding new algorithms, the `swig` wrappers have to be recreated. To this end, `swig` has to be installed and a `pip` package can be created and installed  using
```
make && pip install dSalmon.tar.xz
```
