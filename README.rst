dSalmon
=======

.. image:: https://img.shields.io/github/license/CN-TU/dSalmon.svg
   :target: https://github.com/CN-TU/dSalmon/blob/master/LICENSE
   :alt: License
   
.. image:: https://readthedocs.org/projects/dsalmon/badge/?version=latest
   :target: https://dsalmon.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

dSalmon (**D**\ ata **S**\ tream **A**\ nalysis A\ **l**\ gorith\ **m**\ s f\ **o**\ r the Impatie\ **n**\ t) is a framework for analyzing data streams. Implementation of the core algorithms is done in C++, focusing on superior processing speed and allowing even vast amounts of data to be processed. Python bindings are provided to allow seamless integration in data science development.

Installation
------------
dSalmon can be installed from PyPI using

.. code-block:: sh

    pip3 install dSalmon

or directly from our `GitHub repository <https://github.com/CN-TU/dSalmon>`_:

.. code-block:: sh

    pip3 install git+https://github.com/CN-TU/dSalmon


Outlier Detectors
-----------------
dSalmon provides several algorithms for detecting outliers in data streams. Usage is easiest using the Python interface, which provides an interface similar to the algorithms from scikit-learn. The following example performs outlier detection with a window size of 100 samples.

.. code-block:: python

    from dSalmon import outlier
    import numpy as np
    from sklearn.datasets import fetch_kddcup99
    from sklearn.preprocessing import minmax_scale
    
    # Let scikit-learn fetch and preprocess some stream data
    kddcup = fetch_kddcup99()
    X = minmax_scale(np.delete(kddcup.data, (1,2,3), 1))

    # Perform outlier detection using a Robust Random Cut Forest
    detector = outlier.SWRRCT(window=100)
    outlier_scores = detector.fit_predict(X)
    print ('Top 10 outliers: ', np.argsort(outlier_scores)[-10:])

Individual rows of the passed data are processed sequentially. Hence, while being substantially faster, the above code is equivalent to the following example.

.. code-block:: python

    from dSalmon import outlier
    import numpy as np
    from sklearn.datasets import fetch_kddcup99
    from sklearn.preprocessing import minmax_scale
    
    kddcup = fetch_kddcup99()
    X = minmax_scale(np.delete(kddcup.data, (1,2,3), 1))

    detector = outlier.SWRRCT(window=100)
    outlier_scores = [ detector.fit_predict(row) for row in X ]
    print ('Top 10 outliers: ', np.argsort(outlier_scores)[-10:])

For an overview of provided outlier detection models, consult `dSalmon's documentation <https://dsalmon.readthedocs.io/en/latest/generated/dSalmon.outlier.html>`_.


Obtaining Sliding-Window Statistics
-----------------------------------
Concept drift frequently requires computing statistics based on the most recently observed `N` data samples, since earlier portions of the stream are no longer relevant for the current point in time.

dSalmon provides a `StatisticsTree <https://dsalmon.readthedocs.io/en/latest/generated/dSalmon.trees.html#dSalmon.trees.StatisticsTree>`_, which allows to compute sliding-window statistics efficiently. The following listing provides an example for usage computing the average and 75% percentile of data observed in a sliding window of length 100:

.. code-block:: python

    from dSalmon.trees import StatisticsTree
    import numpy as np

    data = np.random.rand(1000,2)

    tree = StatisticsTree(window=100, what=['average'], quantiles=[0.75])
    stats, sw_counts = tree.fit_query(data)
    print ('Averages:', stats[:,0,:])
    print ('75% percentiles:', stats[:,1,:])

`StatisticsTree <https://dsalmon.readthedocs.io/en/latest/generated/dSalmon.trees.html#dSalmon.trees.StatisticsTree>`_ allows simultaneously querying various statistics. By relying on tree-based methods, time complexity is linear in window length, paving the way for analyzing streams with large memory lengths. 

Stream Scaling
--------------
Performing traditional scaling for streaming data is unrealistic, since in a practical scenario it would involve using data observed in future for scaling. Furthermore, due to concept drift, preprocessing and postprocessing for stream data frequently require scaling values with regard to recently observed values. dSalmon provides tools for these tasks, allowing to perform `z-score scaling <https://dsalmon.readthedocs.io/en/latest/generated/dSalmon.scalers.html#dSalmon.scalers.SWZScoreScaler>`_ and `quantile scaling <https://dsalmon.readthedocs.io/en/latest/generated/dSalmon.scalers.html#dSalmon.scalers.SWQuantileScaler>`_  based on statistics observed in a sliding window. The following example performs outlier detection as demonstrated above, but uses sliding window-based z-score scaling for preprocessing:

.. code-block:: python

    from dSalmon import outlier
    from dSalmon.scalers import SWZScoreScaler
    import numpy as np
    from sklearn.datasets import fetch_kddcup99
    
    # Let scikit-learn fetch and preprocess some stream data
    kddcup = fetch_kddcup99()

    scaler = SWZScoreScaler(window=1000)
    X = scaler.transform(np.delete(kddcup.data, (1,2,3), 1))

    # Omit the first `window` points to avoid transient effects
    X = X[1000:]

    # Perform outlier detection using a Robust Random Cut Forest
    detector = outlier.SWRRCT(window=100)
    outlier_scores = detector.fit_predict(X)
    print ('Top 10 outliers: ', np.argsort(outlier_scores)[-10:])

Efficient Nearest-Neighbor Queries
----------------------------------
dSalmon uses an `M-Tree <https://dsalmon.readthedocs.io/en/latest/generated/dSalmon.trees.html#dSalmon.trees.MTree>`_ for several of its algorithms. An M-Tree is a spatial indexing data structure for metric spaces, allowing fast nearest-neighbor and range queries. The benefit of an M-Tree compared to, e.g., a KD-Tree or Ball-Tree is that insertion, updating and removal of points is fast after having built the tree.

For the development of custom algorithms, an M-Tree interface is provided for Python.
A point within a tree can be accessed either via ``tree[k]`` using the point's key ``k``, or via ``tree.ix[i]`` using the point's index ``i``. Keys can be arbitrary integers and are returned by ``insert()``, ``knn()`` and
``neighbors()``. Indices are integers in the range ``0...len(tree)``, sorted according to the points' keys in ascending order.

KNN queries can be performed using the ``knn()`` function and range queries can be performed using the ``neighbors()`` function.

The following example shows how to modify points within a tree and how to find nearest neighbors.

.. code-block:: python

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


Extending dSalmon
-----------------

dSalmon uses `SWIG <http://www.swig.org/>`_ for generating wrapper code for the C++ core algorithms and instantiates single and double precision floating point variants of each algorithm.

Architecture
^^^^^^^^^^^^

The ``cpp`` folder contains the code for the C++ core algorithms, which might be used directly by C++ projects.

When using dSalmon from Python, the C++ algorithms are wrapped by the interfaces in the SWIG folder. These wrapper functions are translated to a Python interface and have the main purpose of providing an interface which can easily be parsed by SWIG.

Finally, the ``python`` folder contains the Python interface invoking the Python interface provided by SWIG.

Rebuilding
^^^^^^^^^^

When adding new algorithms or modifying the interface, the SWIG wrappers have to be rebuilt. To this end, SWIG has to be installed and a ``pip`` package can be created and installed  using

.. code-block:: sh

    make && pip3 install dSalmon.tar.xz

Acknowledgements
----------------
This work was supported by the project MALware cOmmunication in cRitical Infrastructures (MALORI), funded by the Austrian security research program KIRAS of the Federal Ministry for Agriculture, Regions and Tourism (BMLRT) under grant no. 873511.
