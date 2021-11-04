# Copyright (c) 2020 CN Group, TU Wien
# Released under the GNU Lesser General Public License version 3,
# see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

"""
Indexing structures for fast stream data processing.
"""

import numpy as np
import math
import multiprocessing as mp

from dSalmon import swig as dSalmon_cpp
from dSalmon.util import sanitizeData, sanitizeTimes, lookupDistance


class StatisticsTree(object):
    """
    Indexing structure for computing per-dimension statistics in a sliding
    window.

    This implementation relies on an order statistic tree provided by Boost
    for achieving O(log(window)) time complexity for quantile computation.


    Parameters
    ----------
    window: float
        Window length after which samples will be pruned.

    what: list of strings, optional
        Which statistics to compute. Elements of `what` can be one of
        'sum', 'average', 'squares_sum', 'variance', 'min', 'max'
        or 'median'.

    quantiles: list of floats, optional
        Quantile values to compute in addition to statistics in `what`.
        Elements should be floats in [0,1].

    float_type: np.float32 or np.float64
        The floating point type to use for internal processing.
    """
    def __init__(self, window, what=[], quantiles=[], float_type=np.float64):
        self.window = window
        self.float_type = float_type
        cpp_obj = {np.float32: dSalmon_cpp.StatisticsTree32, np.float64: dSalmon_cpp.StatisticsTree64}[float_type]
        self.cpp_tree = cpp_obj(window)
        if isinstance(what, str):
            what = [ what ]
        self.stats = what
        stat_mapping = {
            'sum': self.cpp_tree.STAT_SUM,
            'average': self.cpp_tree.STAT_AVERAGE,
            'squares_sum': self.cpp_tree.STAT_SQUARES_SUM,
            'variance': self.cpp_tree.STAT_VARIANCE,
            'min': self.cpp_tree.STAT_MIN,
            'max': self.cpp_tree.STAT_MAX,
            'median': self.cpp_tree.STAT_MEDIAN,
        }
        self.mapped_stats = [ stat_mapping[x] for x in what ]
        self.quantiles = np.array(quantiles, dtype=self.float_type)
        assert (0 <= self.quantiles).all() and (self.quantiles <= 1).all()
        self.last_time = 0

    def fit_query(self, X, times=None):
        """
        Process next chunk of data.

        Parameters
        ----------
        X: ndarray, shape (n_samples, n_features)
            The input data.
            
        times: ndarray, shape (n_samples,), optional
            Timestamps for input data. If None, timestamps are linearly
            increased for each sample.

        Returns
        -------
        statistics: ndarray, shape (n_samples, n_statistics, n_features)
            The computed statistics. Statistics for row `i` are evaluated
            after adding row `i` to the sliding window.
            Here, `n_statistics` = `len(what)` + `len(quantiles)`.

        counts: ndarray, shape (n_samples)
            The lengths of the sliding window after processing each row
            of `X`.
        """
        X = sanitizeData(X, self.float_type)
        times = sanitizeTimes(times, X.shape[0], self.last_time, self.float_type)
        result = np.empty((X.shape[0], len(self.mapped_stats) + len(self.quantiles), X.shape[1]), dtype=self.float_type)
        counts = np.empty(X.shape[0], dtype=np.int64)
        self.cpp_tree.fit_query(X, times, self.mapped_stats, self.quantiles, result, counts)
        self.last_time = times[-1]
        return result, counts


class _MTreeByIndexLookup(object):
    def __init__(self, tree_obj):
        self.tree_obj = tree_obj
    
    def __len__(self):
        return len(self.tree_obj)
        
    def __getitem__(self, indices):
        sel, result_length = self.tree_obj._byIndexSelector(indices)
                
        # TODO: should do something if both sel and tree are empty
        points = np.empty((result_length, self.tree_obj.dimension), dtype=self.tree_obj.float_type)
        self.tree_obj.cpp_tree.getPoints(sel, points)
        return points
        
    def __setitem__(self, indices, data):
        data = sanitizeData(data, self.tree_obj.float_type)
        assert self.tree_obj.dimension == data.shape[1]
        
        if isinstance(indices, slice):
            start,stop,stride = indices.indices(len(self))
            assert (stop-start)//stride == data.shape[0], "Slice doesn't match assigned size"
            self.tree_obj.cpp_tree.updateByIndexSlice(start, stop, stride, data)
        else:
            indices = np.array(indices, dtype=np.int64)
            if len(indices.shape) == 0:
                indices = indices[None]
            assert len(indices.shape) == 1
            indices[indices<0] += len(self)
            assert (0<=indices).all() and (indices<len(self)).all()
            assert indices.shape[0] == data.shape[0], "Slice doesn't match assigned size"
            self.tree_obj.cpp_tree.updateByIndex(indices, data)
        
    def __delitem__(self, indices):
        sel,_ = self.tree_obj._byIndexSelector(indices)
        self.tree_obj.cpp_tree.remove(sel)


class MTree(object):
    """
    M-Tree efficient nearest-neighbor search in metric spaces.
    
    A point within a tree can be accessed either via tree[k] using the
    point's key k, or via tree.ix[i] using the point's index i.
    Keys can be arbitrary integers and are returned by the insert, knn and
    neighbors functions. Indices are integers in the range 0...len(tree), sorted
    according to the points' keys in ascending order.

    Parameters
    ----------
    metric: string
        Which distance metric to use. Currently supported metrics
        include 'chebyshev', 'cityblock', 'euclidean' and
        'minkowsi'.

    metric_params: dict
        Parameters passed to the metric. Minkowsi distance requires
        setting an integer `p` parameter.

    float_type: np.float32 or np.float64
        Which floating point type to use for internal processing.
        
    min_node_size: int
        The minimum number of children in tree nodes. Different
        parametrizations for min_node_size are guaranteed to
        return identical results.
        
    max_node_size: int
        The maximum number of children in tree nodes. Different
        parametrizations for max_node_size are guaranteed to
        return identical results.
        
    split_sampling: int
        The number of combinations to try when splitting a node.
        Different parametrizations for split_sampling are guaranteed
        to return identical results.

    insert_jobs: int
        The number of additional threads to spawn for tree insertions.
        Since insertions can only partially be parallelized, using 
        too many threads can harm performance.

    query_jobs: int
        The number of threads to use for range- and knn-queries.
    """
    
    def __init__(self, metric='euclidean', metric_params=None, float_type=np.float64, min_node_size=5, max_node_size=100, split_sampling=20, insert_jobs=2, query_jobs=-1, **kwargs):
        assert float_type in [np.float32, np.float64]
        assert min_node_size * 2 < max_node_size
        self.float_type = float_type
        distance_function = lookupDistance(metric, float_type, **(metric_params or {}))
        insert_jobs = mp.cpu_count() if insert_jobs==-1 else insert_jobs
        query_jobs = mp.cpu_count() if query_jobs==-1 else query_jobs
        cpp_obj = {np.float32: dSalmon_cpp.MTree32, np.float64: dSalmon_cpp.MTree64}[float_type]
        self.cpp_tree = cpp_obj(distance_function, int(min_node_size), int(max_node_size), int(split_sampling), int(insert_jobs), int(query_jobs))
        self.ix = _MTreeByIndexLookup(self)
        self.float_type = float_type
        if float_type == np.float32:
            self.selector_factory = dSalmon_cpp.MTreeSelector32
            self.range_query_factory = dSalmon_cpp.MTreeRangeQuery32
            self.knn_query_factory = dSalmon_cpp.MTreeKnnQuery32
        else:
            self.selector_factory = dSalmon_cpp.MTreeSelector64
            self.range_query_factory = dSalmon_cpp.MTreeRangeQuery64
            self.knn_query_factory = dSalmon_cpp.MTreeKnnQuery64
        
    def _byKeySelector(self, keys, ignore_missing=False):
        assert not isinstance(keys, slice), 'Slice access is not supported for this object. Use .ix[] instead.'
        keys = np.array(keys, dtype=np.int64)
        if len(keys.shape) == 0:
            keys = keys[None]
        assert len(keys.shape) == 1
        sel = self.selector_factory(self.cpp_tree, keys, False)
        if not ignore_missing and not sel.allOk():
            found = np.empty(keys.shape[0], dtype=np.uint8)
            sel.getFoundMask(found)
            raise IndexError('Keys ' + str(keys[found==0].tolist()) + ' invalid')
        return sel, len(keys)
        
    def _byIndexSelector(self, indices, ignore_missing=False):
        if isinstance(indices, slice):
            start,stop,stride = indices.indices(len(self))
            result_length = (stop-start)//stride
            sel = self.selector_factory(self.cpp_tree, start, stop, stride)
            assert sel.allOk()
        else:
            indices = np.array(indices, dtype=np.int64)
            if len(indices.shape) == 0:
                indices = indices[None]
            assert len(indices.shape) == 1
            indices[indices<0] += len(self)
            assert (0<=indices).all() and (indices<len(self)).all()
            result_length = len(indices)
            sel = self.selector_factory(self.cpp_tree, indices, True)
            if not ignore_missing and not sel.allOk():
                found = np.empty(len(indices), dtype=np.uint8)
                sel.getFoundMask(found)
                raise IndexError('Indices ' + str(indices[found==0].tolist()) + ' invalid')
        return sel, result_length
        
    def insert(self, data):
        """
        Insert points and return indices.
        
        Parameters
        ----------
        data: ndarray, shape (n_samples, n_features)
            The data to be inserted.
            
        Returns
        -------
        indices: ndarray, shape (n_samples,)
            The indices assigned to the newly inserted data points.
        """
        data = sanitizeData(data, self.float_type)
        assert self.dimension in [-1, data.shape[1]]
        indices = np.empty(data.shape[0], dtype=np.int64)
        self.cpp_tree.insert(data, indices)
        return indices
        
    def remove(self, keys):
        """Remove points identified by keys, skipping non-existent entries.
        
        Parameters
        ---------------
        keys: ndarray, shape (n_samples,)
            Indices of the data points to be removed.
            
        Returns
        ---------------
        found: ndarray, shape (n_samples,)
            Boolean array indicating whether the removal was successful.
        """
        sel, result_length = self._byKeySelector(keys, ignore_missing=True)
        if self.dimension == -1:
            return np.zeros(len(keys), dtype=bool)
        found = np.empty(len(keys), dtype=np.uint8)
        self.cpp_tree.remove(sel)
        sel.getFoundMask(found)
        return found != 0

    def neighbors(self, data, radius):
        """
        Return all points within a given radius.
        
        Parameters
        ---------------
        data: ndarray, shape (n_samples, n_features)
            Points for which the range query should be performed.
            
        radius: double
            Radius for the search.
            
        Returns
        ---------------
        keys: ndarray, shape (n_total_neighbors,)
            Concatenation of keys of neighbors within radius. 
            
        distances: ndarray, shape (n_total_neighbors,)
            Concatenation of distances of neighbors to the respective
            query points.
        
        lengths: ndarray, shape (n_samples,)
            The number of neighbors returned for each point, so that
            sum(length) == n_total_neighbors.
        """
        data = sanitizeData(data, self.float_type)
        assert self.dimension == data.shape[1]
        query = self.range_query_factory(self.cpp_tree, data, radius)
        total_length = query.resultTotalLength()
        keys = np.empty(total_length, dtype=np.int64)
        distances = np.empty(total_length, dtype=self.float_type)
        lengths = np.empty(data.shape[0], dtype=np.int32)
        query.result(keys, distances)
        query.resultLengths(lengths)
        return keys, distances, lengths
    
    def knn(self, data, k=1, sort=True, min_radius=0, max_radius=math.inf, reverse=False, extend_for_ties=False):
        """
        Find the k nearest neighbors of points.
        
        Parameters
        ---------------
        data: ndarray, shape (n_samples, n_features)
            Points for which to perform a knn query.
            
        k: int
            Number of nearest neighbors to consider.
                        
        sort: bool
            Whether the returned points should be sorted by distance.
                        
        min_radius: double
            Minimum distance for returned neighbor points.    
                                
        max_radius: double
            Maximum distance for returned neighbor points.
                                            
        reverse: bool
            If reverse == True, return the k most distant points instead
            of the k nearest neighbors.
            
        extend_for_ties: bool
            Whether in the case of ties more than k points should
            be returned.
            
        Returns
        ---------------
        keys: ndarray, shape (n_total_neighbors,)
            Concatenation of keys of found neighbors. 
            
        distances: ndarray, shape (n_total_neighbors,)
            Concatenation of distances of neighbors to the respective
            query points.
        
        lengths: ndarray, shape (n_samples,)
            The number of neighbors returned for each point, so that
            sum(length) == n_total_neighbors.
        """
        data = sanitizeData(data, self.float_type)
        assert self.dimension == data.shape[1]
        query = self.knn_query_factory(self.cpp_tree, data, int(k), sort,
                                       min_radius, max_radius, reverse,
                                       extend_for_ties)
        total_length = query.resultTotalLength()
        keys = np.empty(total_length, dtype=np.int64)
        distances = np.empty(total_length, dtype=self.float_type)
        lengths = np.empty(data.shape[0], dtype=np.int32)
        query.result(keys, distances)
        query.resultLengths(lengths)
        return keys, distances, lengths

    def clear(self):
        """Remove all points from the tree."""
        self.cpp_tree.clear()
        
    def __len__(self):
        return self.cpp_tree.size()
        
    @property
    def dimension(self):
        return self.cpp_tree.dataDimension()
        
    def get_points(self, keys):
        """
        Retrieve points by key, skipping non-existent entries.
        
        Parameters
        ---------------
        keys: ndarray, shape (n_samples,)
            Keys for points to query as returned by insert().
            
        Returns
        ---------------
        points: ndarray, shape (n_samples, n_features)
            Coordinates of queried points or all-zero vectors if 
            points were not found in the tree. 
            
        found: ndarray, shape (n_samples,)
            Whether the respective keys were found in the tree.
        """
        sel, result_length = self._byKeySelector(keys, ignore_missing=True)
        if self.dimension == -1:
            return np.zeros((0,0), dtype=self.float_type), np.zeros(len(keys), dtype=bool)
        points = np.empty((len(keys), self.dimension), dtype=self.float_type)
        found = np.empty(len(keys), dtype=np.uint8)
        self.cpp_tree.getPoints(sel, points)
        sel.getFoundMask(found)
        return points, found!=0
        
    def __getitem__(self, keys):
        sel, result_length = self._byKeySelector(keys)
        # TODO: should handle dimension if both key and tree are empty
        points = np.empty((result_length, self.dimension), dtype=self.float_type)
        self.cpp_tree.getPoints(sel, points)
        return points
        
    def __setitem__(self, keys, data):
        assert not isinstance(keys, slice), 'Slice access is not supported for this object. Use .ix[] instead.'
        data = sanitizeData(data, self.float_type)
        keys = np.array(keys, dtype=np.int64)
        if len(keys.shape) == 0:
            keys = keys[None]
        assert len(keys.shape) == 1
        assert self.dimension in [-1, data.shape[1]]
        assert len(keys) == data.shape[0]
        
        self.cpp_tree.update(keys, data)
        
    def __delitem__(self, keys):
        sel,_ = self._byKeySelector(keys)
        self.cpp_tree.remove(sel)
        
    def itok(self, indices=None):
        """
        Map indices to keys.
        
        Parameters
        ----------
        indices: ndarray or slice, optional
            Indices or slice for which to return keys. If None,
            all keys are returned.
            
        Returns
        -------
        keys: ndarray
            The requested keys as numpy array.
        """
        if indices is None:
            indices = slice(None)
        sel, result_length = self._byIndexSelector(indices)
        keys = np.empty(result_length, dtype=np.int64)
        self.cpp_tree.keys(sel, keys)
        return keys
        
    def ktoi(self, keys):
        """
        Map keys to indices.
        
        Parameters
        ----------
        keys: ndarray
            Keys for which to return indices.
            
        Returns
        -------
        indices: ndarray
            The requested indices as numpy array.
        """
        sel, result_length = self._byKeySelector(keys)
        indices = np.empty(result_length, dtype=np.int64)
        self.cpp_tree.indices(sel, indices)
        return indices
        
    def copy(self):
        """Return a copy of the tree."""
        new_tree = MTree(float_type=self.float_type)
        new_tree.cpp_tree.unserialize(self.cpp_tree.serialize())
        return new_tree

    def __getstate__(self):
        return {
            'version': 0,
            'float_type': self.float_type,
            'data': self.cpp_tree.serialize()
        }

    def __setstate__(self, state):
        assert state['version'] == 0, 'Unknown serialization version'
        self.__init__(float_type=state['float_type'])
        # TODO: is distance function properly serialized?
        if not self.cpp_tree.unserialize(state['data']):
            raise ValueError('Unsupported serialization version')

