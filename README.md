# dSalmon

[![License](https://img.shields.io/github/license/CN-TU/dSalmon.svg)](https://github.com/CN-TU/dSalmon/blob/master/LICENSE)

dSalmon (Data Stream Analysis Algorithms for the Impatient) is a framework for analyzing data streams. Implementation of the core algorithms is done in C++, focusing on superior processing speed and allowing even vast amounts of data to be processed. Python bindings are provided to allow seamless integration in data science development.

## Installation
dSalmon can be installed using `pip` by running
```
pip3 install git+https://github.com/CN-TU/dSalmon
```

## Outlier Detectors
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

## M-Tree usage
dSalmon uses an M-Tree for several of its algorithms. An M-Tree is a spatial indexing data structure for metric spaces, allowing fast nearest-neighbor and range queries. The benefit of an M-Tree compared to, e.g., a KD-Tree or Ball-Tree is that insertion, updating and removal of points is fast after having built the tree.

For the development of custom algorithms, an M-Tree interface is provided for Python.
A point within a tree can be accessed either via `tree[k]` using the point's key `k`, or via `tree.ix[i]` using the point's index `i`. Keys can be arbitrary integers and are returned by the `insert`, `knn` and
`neighbors` functions. Indices are integers in the range `0...len(tree)`, sorted according to the points' keys in ascending order.

Knn queries can be performed using the `knn()` function and range queries can be performed using the `neighbors()` function.

The following example shows how to modify points within a tree and how to find nearest neighbors.
```python
from dSalmon.trees import MTree
import numpy as np

tree = MTree()

# insert a point [1,2,3,4] with key 5
tree[5] = [1,2,3,4]

# insert some random test data
X = np.random.rand(1000,4)
inserted_keys = tree.insert(X)

# delete every second point
del tree.ix[::2]

# Set the coordinates of the point with the lowest key
tree.ix[0] = [0,0,0,0]

# find the 3 nearest neighbors to [0.5, 0.5, 0.5, 0.5]
neighbor_keys, neighbor_distances, _ = tree.knn([.5,.5,.5,.5], k=3)
print ('Neighbor keys:', neighbor_keys)
print ('Neighbor distances:', neighbor_distances)

# find all neighbors to [0.5, 0.5, 0.5, 0.5] within a radius of 0.2
neighbor_keys, neighbor_distances, _ = tree.neighbors([.5,.5,.5,.5], radius=0.2)
print ('Neighbor keys:', neighbor_keys)
print ('Neighbor distances:', neighbor_distances)
```


## Extending dSalmon
dSalmon uses `swig` for generating wrapper code for the C++ core algorithms and instantiates single and double precision floating point variants of each algorithm.

### Architecture
The `cpp` folder contains the code for the C++ core algorithms, which might be used directly by C++ projects.

When using dSalmon from Python, the C++ algorithms are wrapped by the interfaces in the `swig` folder. These wrapper functions are translated to a Python interface and have the main purpose of providing an interface which can easily be parsed by `swig`.

Finally, the `python` folder contains the Python interface invoking the Python interface provided by `swig`.

### Rebuilding
When adding new algorithms or modifying the interface, the `swig` wrappers have to be rebuilt. To this end, `swig` has to be installed and a `pip` package can be created and installed  using
```
make && pip3 install dSalmon.tar.xz
```
