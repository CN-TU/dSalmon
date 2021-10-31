# Copyright (c) 2020 CN Group, TU Wien
# Released under the GNU Lesser General Public License version 3,
# see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

"""
Streaming outlier detection models.
"""

import numpy as np
import multiprocessing as mp

from dSalmon import swig as dSalmon_cpp
from dSalmon import projection
from dSalmon.util import sanitizeData, sanitizeTimes, lookupDistance

class OutlierDetector(object):
    """
    Base class for outlier detectors.
    """
    
    def _init_model(self, p):
        pass

    def get_params(self, deep=True):
        """
        Return the used algorithm parameters as dictionary.

        Parameters
        ----------
        deep: bool, default=True
            Ignored. Only for compatibility with scikit-learn.

        Returns
        -------
        params: dict
            Dictionary of parameters.
        """
        return self.params

    def set_params(self, **params):
        """
        Reset the model and set the parameters in accordance to the
        supplied dictionary.

        Parameters
        ----------
        **params: dict
            Dictionary of parameters.
        """
        p = self.params.copy()
        for key in params:
            assert key in p, 'Unknown parameter: %s' % key
            p[key] = params[key]
        self._init_model(p)

    def fit(self, X, times=None):
        """
        Process next chunk of data without returning outlier scores.
        
        Parameters
        ----------
        X: ndarray, shape (n_samples, n_features)
            The input data.
            
        times: ndarray, shape (n_samples,), optional
            Timestamps for input data. If None,
            timestamps are linearly increased for
            each sample. 
        """
        # In most cases, fitting isn't any faster than additionally
        # performing outlier scoring. We override this method only
        # when it yields faster processing.
        self.fit_predict(X, times)

    def _process_data(self, data):
        data = sanitizeData(data, self.params['float_type'])
        assert self.dimension == -1 or data.shape[1] == self.dimension
        self.dimension = data.shape[1]
        return data

    def _process_times(self, data, times):
        times = sanitizeTimes(times, data.shape[0], self.last_time, self.params['float_type'])
        self.last_time = times[-1]
        return times


class SWDBOR(OutlierDetector):
    """
    Distance based outlier detection by radius.

    When setting a threshold for the returned outlier scores to tranform
    outlier scores into binary labels, results coincide with 
    ExactStorm :cite:p:`Angiulli2007`, AbstractC :cite:p:`Yang2009`
    or the COD family :cite:p:`Kontaki2011`.
    
    Parameters
    ----------
    window: float
        Window length after which samples will be pruned.
        
    radius: float
        Radius for classification as neighbor.

    metric: string
        Which distance metric to use. Currently supported metrics
        include 'chebyshev', 'cityblock', 'euclidean' and
        'minkowsi'.

    metric_params: dict
        Parameters passed to the metric. Minkowsi distance requires
        setting an integer `p` parameter.

    float_type: np.float32 or np.float64
        The floating point type to use for internal processing.
        
    min_node_size: int, optional (default=5)
        Smallest possible size for M-Tree nodes. min_node_size
        is guaranteed to leave results unaffected.
        
    max_node_size: int, optional (default=20)
        Largest possible size for M-Tree nodes. max_node_size
        is guaranteed to leave results unaffected.
        
    split_sampling: int, optional (default=5)
        The number of key combinations to try when splitting M-Tree 
        routing nodes. split_sampling is guaranteed to leave results
        unaffected.
    """
    
    def __init__(self, window, radius, metric='euclidean', metric_params=None, float_type=np.float64, min_node_size=5, max_node_size=100, split_sampling=20):
        self.params = { k: v for k, v in locals().items() if k != 'self' }
        self._init_model(self.params)

    def _init_model(self, p):
        assert p['float_type'] in [np.float32, np.float64]
        assert p['max_node_size'] > 2 * p['min_node_size'], 'max_node_size must be > 2 * min_node_size'
        assert p['min_node_size'] > 0
        assert p['window'] > 0
        assert p['radius'] > 0
        distance_function = lookupDistance(p['metric'], p['float_type'], **(p['metric_params'] or {}))
        cpp_obj = {np.float32: dSalmon_cpp.DBOR32, np.float64: dSalmon_cpp.DBOR64}[p['float_type']]
        self.model = cpp_obj(p['window'], p['radius'], distance_function, p['min_node_size'],
                             p['max_node_size'], p['split_sampling'])
        self.last_time = 0
        self.dimension = -1
        self.params = p

    def fit_predict(self, X, times=None):
        """
        Process next chunk of data.
        
        Parameters
        ----------
        X: ndarray, shape (n_samples, n_features)
            The input data.
            
        times: ndarray, shape (n_samples,), optional
            Timestamps for input data. If None,
            timestamps are linearly increased for
            each sample. 
        
        Returns    
        -------
        y: ndarray, shape (n_samples,)
            Outlier scores for provided input data.
        """
        X = self._process_data(X)
        times = self._process_times(X, times)
        scores = np.empty(X.shape[0], dtype=self.params['float_type'])
        self.model.fit_predict(X, scores, times)
        return scores
        
    def window_size(self):
        """Return the number of samples in the sliding window."""
        return self.model.window_size()
        
    def get_window(self):
        """
        Return samples in the current window.
        
        Returns
        -------
        data: ndarray, shape (n_samples, n_features)
            Samples in the current window.
            
        times: ndarray, shape (n_samples,)
            Expiry times of samples in the current window.
            
        neighbors: ndarray, shape (n_samples)
            Number of neighbors of samples in the current
            window.
        """
        if self.dimension == -1:
            return np.zeros([0], dtype=self.params['float_type']), np.zeros([0], dtype=self.params['float_type']), np.zeros([0], dtype=np.int32)
        window_size = self.model.window_size()
        data = np.empty([window_size, self.dimension], dtype=self.params['float_type'])
        times = np.empty(window_size, dtype=self.params['float_type'])
        neighbors = np.empty(window_size, dtype=np.int32)
        self.model.get_window(data, times, neighbors)
        return data, times, neighbors


class SWKNN(OutlierDetector):
    """
    Distance based outlier detection by k nearest neighbors.

    When setting a threshold for the returned outlier scores to tranform
    outlier scores into binary labels, results coincide with 
    ExactStorm :cite:p:`Angiulli2007`, AbstractC :cite:p:`Yang2009`
    or the COD family :cite:p:`Kontaki2011`.

    Parameters
    ----------
    window: float
        Window length after which samples will be pruned.
        
    k: int
        Number of nearest neighbors to consider for outlier
        scoring.
        
    k_is_max: bool (default=False)
        Whether scores should be returned for all neighbor values
        up to the provided k.
        Grid search for the optimal k can be performed by setting
        k_is_max=True.

    metric: string
        Which distance metric to use. Currently supported metrics
        include 'chebyshev', 'cityblock', 'euclidean' and
        'minkowsi'.

    metric_params: dict
        Parameters passed to the metric. Minkowsi distance requires
        setting an integer `p` parameter.

    float_type: np.float32 or np.float64
        The floating point type to use for internal processing.
        
    min_node_size: int, optional (default=5)
        Smallest possible size for M-Tree nodes. min_node_size
        is guaranteed to leave results unaffected.
        
    max_node_size: int, optional (default=20)
        Largest possible size for M-Tree nodes. max_node_size
        is guaranteed to leave results unaffected.

    split_sampling: int, optional (default=5)
        The number of key combinations to try when splitting M-Tree 
        routing nodes. split_sampling is guaranteed to leave results
        unaffected.
    """
    
    def __init__(self, window, k, k_is_max=False, metric='euclidean', metric_params=None, float_type=np.float64, min_node_size = 5, max_node_size = 100, split_sampling = 20):
        self.params = { k: v for k, v in locals().items() if k != 'self' }
        self._init_model(self.params)

    def _init_model(self, p):
        assert p['float_type'] in [np.float32, np.float64]
        assert p['max_node_size'] > 2 * p['min_node_size'], 'max_node_size must be > 2 * min_node_size'
        assert p['min_node_size'] > 0
        assert p['window'] > 0
        assert p['k'] > 0
        distance_function = lookupDistance(p['metric'], p['float_type'], **(p['metric_params'] or {}))
        cpp_obj = {np.float32: dSalmon_cpp.SWKNN32, np.float64: dSalmon_cpp.SWKNN64}[p['float_type']]
        self.model = cpp_obj(p['window'], p['k'], distance_function, p['min_node_size'],
                             p['max_node_size'], p['split_sampling'])
        self.last_time = 0
        self.dimension = -1

    def fit(self, X, times=None):
        """
        Process next chunk of data without returning outlier scores.
        
        Parameters
        ----------
        X: ndarray, shape (n_samples, n_features)
            The input data.
            
        times: ndarray, shape (n_samples,), optional
            Timestamps for input data. If None,
            timestamps are linearly increased for
            each sample. 
        """
        X = self._process_data(X)
        times = self._process_times(X, times)
        self.model.fit(X, times)

    def fit_predict(self, X, times=None):
        """
        Process next chunk of data.
        
        Parameters
        ----------
        X: ndarray, shape (n_samples, n_features)
            The input data.
            
        times: ndarray, shape (n_samples,), optional
            Timestamps for input data. If None,
            timestamps are linearly increased for
            each sample. 
        
        Returns    
        -------
        y: ndarray, shape (n_samples,) or (n_samples,k)
            Outlier scores for provided input data.
        """
        X = self._process_data(X)
        times = self._process_times(X, times)
        if self.params['k_is_max']:
            scores = np.empty([X.shape[0], self.params['k']], dtype=self.params['float_type'])
            self.model.fit_predict_with_neighbors(X, scores, times)
        else:
            scores = np.empty(X.shape[0], dtype=self.params['float_type'])
            self.model.fit_predict(X, scores, times)
        return scores

    def window_size(self):
        """Return the number of samples in the sliding window."""
        return self.model.window_size()
        
    def get_window(self):
        """
        Return samples in the current window.
        
        Returns
        -------
        data: ndarray, shape (n_samples, n_features)
            Samples in the current window.
            
        times: ndarray, shape (n_samples,)
            Expiry times of samples in the current window.
        """
        if self.dimension == -1:
            return np.zeros([0], dtype=self.params['float_type']), np.zeros([0], dtype=self.params['float_type']), np.zeros([0], dtype=np.int32)
        window_size = self.model.window_size()
        data = np.empty([window_size, self.dimension], dtype=self.params['float_type'])
        times = np.empty(window_size, dtype=self.params['float_type'])
        self.model.get_window(data, times)
        return data, times

        
class SWLOF(OutlierDetector):
    """
    Local Outlier Factor :cite:p:`Breunig2000` within a sliding window.

    Parameters
    ----------
    window: float
        Window length after which samples will be pruned.
        
    k: int
        Number of nearest neighbors to consider for outlier
        scoring.
        
    simplified: bool (default=False)
        Whether to use simplified LOF.
        
    k_is_max: bool (default=False)
        Whether scores should be returned for all neighbor values
        up to the provided k.
        Grid search for the optimal k can be performed by setting
        k_is_max=True.

    metric: string
        Which distance metric to use. Currently supported metrics
        include 'chebyshev', 'cityblock', 'euclidean' and
        'minkowsi'.

    metric_params: dict
        Parameters passed to the metric. Minkowsi distance requires
        setting an integer `p` parameter.

    float_type: np.float32 or np.float64
        The floating point type to use for internal processing.
        
    min_node_size: int, optional (default=5)
        Smallest possible size for M-Tree nodes. min_node_size
        is guaranteed to leave results unaffected.
        
    max_node_size: int, optional (default=20)
        Largest possible size for M-Tree nodes. max_node_size
        is guaranteed to leave results unaffected.
        
    split_sampling: int, optional (default=5)
        The number of key combinations to try when splitting M-Tree 
        routing nodes. split_sampling is guaranteed to leave results
        unaffected.
    """
    
    def __init__(self, window, k, simplified=False, k_is_max=False, metric='euclidean', metric_params=None, float_type=np.float64, min_node_size=5, max_node_size=100, split_sampling=20):
        self.params = { k: v for k, v in locals().items() if k != 'self' }
        self._init_model(self.params)

    def _init_model(self, p):
        assert p['float_type'] in [np.float32, np.float64]
        assert p['max_node_size'] > 2 * p['min_node_size'], 'max_node_size must be > 2 * min_node_size'
        assert p['min_node_size'] > 0
        assert p['window'] > 0
        assert p['k'] > 0
        distance_function = lookupDistance(p['metric'], p['float_type'], **(p['metric_params'] or {}))
        cpp_obj = {np.float32: dSalmon_cpp.SWLOF32, np.float64: dSalmon_cpp.SWLOF64}[p['float_type']]
        self.model = cpp_obj(p['window'], p['k'], p['simplified'], distance_function,
                             p['min_node_size'], p['max_node_size'], p['split_sampling'])
        self.last_time = 0
        self.dimension = -1
        
    def fit(self, data, times=None):
        """
        Process next chunk of data without returning outlier scores.
        
        Parameters
        ----------
        X: ndarray, shape (n_samples, n_features)
            The input data.
            
        times: ndarray, shape (n_samples,), optional
            Timestamps for input data. If None,
            timestamps are linearly increased for
            each sample. 
        """
        data = self._process_data(data)
        times = self._process_times(data, times)
        self.model.fit(data, times)
        
    def fit_predict(self, X, times=None):
        """
        Process next chunk of data.
        
        Parameters
        ----------
        X: ndarray, shape (n_samples, n_features)
            The input data.
            
        times: ndarray, shape (n_samples,), optional
            Timestamps for input data. If None,
            timestamps are linearly increased for
            each sample. 
        
        Returns    
        -------
        y: ndarray, shape (n_samples,) or (n_samples,k)
            Outlier scores for provided input data.
        """
        X = self._process_data(X)
        times = self._process_times(X, times)
        scores = np.empty([X.shape[0], self.params['k']], dtype=self.params['float_type'])
        self.model.fit_predict(X, scores, times)
        return scores if self.params['k_is_max'] else scores[:,-1]
        
    def window_size(self):
        """Return the number of samples in the sliding window."""
        return self.model.window_size()
        
    def get_window(self):
        """
        Return samples in the current window.
        
        Returns
        -------
        data: ndarray, shape (n_samples, n_features)
            Samples in the current window.
            
        times: ndarray, shape (n_samples,)
            Expiry times of samples in the current window.
        """
        if self.dimension == -1:
            return np.zeros([0], dtype=self.params['float_type']), np.zeros([0], dtype=self.params['float_type']), np.zeros([0], dtype=np.int32)
        window_size = self.model.window_size()
        data = np.empty([window_size, self.dimension], dtype=self.params['float_type'])
        times = np.empty(window_size, dtype=self.params['float_type'])
        self.model.get_window(data, times)
        return data, times

        
class SDOstream(OutlierDetector):
    """
    Streaming outlier detection based on Sparse Data Observers :cite:p:`Hartl2019`.
    
    Parameters
    ----------
    k: int
        Number of observers to use.
        
    T: int
        Characteristic time for the model.
        Increasing T makes the model adjust slower, decreasing T
        makes it adjust quicker.
        
    qv: float, optional (default=0.3)
        Ratio of unused observers due to model cleaning.
        
    x: int (default=6)
        Number of nearest observers to consider for outlier scoring
        and model cleaning.

    metric: string
        Which distance metric to use. Currently supported metrics
        include 'chebyshev', 'cityblock', 'euclidean' and
        'minkowsi'.

    metric_params: dict
        Parameters passed to the metric. Minkowsi distance requires
        setting an integer `p` parameter.

    float_type: np.float32 or np.float64
        The floating point type to use for internal processing.

    seed: int (default=0)
        Random seed to use.
        
    return_sampling: bool (default=False)
        Also return whether a data point was adopted as observer.
    """

    def __init__(self, k, T, qv=0.3, x=6, metric='euclidean', metric_params=None, float_type=np.float64, seed=0, return_sampling=False):
        self.params = { k: v for k, v in locals().items() if k != 'self' }
        self._init_model(self.params)

    def _init_model(self, p):
        assert p['float_type'] in [np.float32, np.float64]
        assert 0 <= p['qv'] < 1, 'qv must be in [0,1)'
        assert p['x'] > 0, 'x must be > 0'
        assert p['k'] > 0, 'k must be > 0'
        assert p['T'] > 0, 'T must be > 0'
        distance_function = lookupDistance(p['metric'], p['float_type'], **(p['metric_params'] or {}))
        cpp_obj = {np.float32: dSalmon_cpp.SDOstream32, np.float64: dSalmon_cpp.SDOstream64}[p['float_type']]
        self.model = cpp_obj(p['k'], p['T'], p['qv'], p['x'], distance_function, p['seed'])
        self.last_time = 0
        self.dimension = -1
        
    def fit_predict(self, X, times=None):
        """
        Process next chunk of data.
        
        Parameters
        ----------
        X: ndarray, shape (n_samples, n_features)
            The input data.
            
        times: ndarray, shape (n_samples,), optional
            Timestamps for input data. If None,
            timestamps are linearly increased for
            each sample. 
        
        Returns    
        -------
        y: ndarray, shape (n_samples,)
            Outlier scores for provided input data.
        """
        X = self._process_data(X)
        times = self._process_times(X, times)
        scores = np.empty(X.shape[0], dtype=self.params['float_type'])
        if self.params['return_sampling']:
            sampling = np.empty(X.shape[0], dtype=np.int32)
            self.model.fit_predict_with_sampling(X, scores, times, sampling)
            return scores, sampling
        else:
            self.model.fit_predict(X, scores, times)
            return scores
        
    def observer_count(self):
        """Return the current number of observers."""
        return self.model.observer_count()
        
    def get_observers(self, time=None):
        """
        Return observer data.
        
        Returns    
        -------
        data: ndarray, shape (n_observers, n_features)
            Sample used as observer.
            
        observations: ndarray, shape (n_observers,)
            Exponential moving average of observations.
            
        av_observations: ndarray, shape (n_observers,)
            Exponential moving average of observations
            normalized according to the theoretical maximum.
        """
        if time is None:
            time = self.last_time
        observer_cnt = self.model.observer_count()
        if observer_cnt == 0:
            return np.zeros([0], dtype=self.params['float_type']), np.zeros([0], dtype=self.params['float_type']), np.zeros([0], dtype=self.params['float_type'])
        data = np.empty([observer_cnt, self.dimension], dtype=self.params['float_type'])
        observations = np.empty(observer_cnt, dtype=self.params['float_type'])
        av_observations = np.empty(observer_cnt, dtype=self.params['float_type'])
        self.model.get_observers(data, observations, av_observations, self.params['float_type'](time))
        return data, observations, av_observations


class SWRRCT(OutlierDetector):
    """
    Robust Random Cut Forest :cite:p:`Guha16`.
    
    Parameters
    ----------
    window: float
        Window length after which samples will be pruned.

    n_estimators: int
        Number of trees in the ensemble.

    float_type: np.float32 or np.float64
        The floating point type to use for internal processing.
        
    seed: int
        Random seed for tree construction.

    n_jobs: int
        Number of threads to use for processing trees.
        Pass -1 to use as many jobs as there are CPU cores.
    """
    
    def __init__(self, window, n_estimators = 10, float_type=np.float64, seed=0, n_jobs=-1):
        self.params = { k: v for k, v in locals().items() if k != 'self' }
        self._init_model(self.params)

    def _init_model(self, p):
        assert p['float_type'] in [np.float32, np.float64]
        assert p['n_estimators'] > 0
        assert p['window'] > 0
        cpp_obj = {np.float32: dSalmon_cpp.RRCT32, np.float64: dSalmon_cpp.RRCT64}[p['float_type']]
        self.model = cpp_obj(p['n_estimators'], p['window'], p['seed'], mp.cpu_count() if p['n_jobs']==-1 else p['n_jobs'])
        self.last_time = 0
        self.dimension = -1

    def fit(self, X, times=None):
        """
        Process next chunk of data without returning outlier scores.
        
        Parameters
        ----------
        X: ndarray, shape (n_samples, n_features)
            The input data.
            
        times: ndarray, shape (n_samples,), optional
            Timestamps for input data. If None,
            timestamps are linearly increased for
            each sample. 
        """
        X = self._process_data(X)
        times = self._process_times(X, times)
        self.model.fit(X, times)

    def fit_predict(self, X, times=None):
        """
        Process next chunk of data.
        
        Parameters
        ----------
        X: ndarray, shape (n_samples, n_features)
            The input data.
            
        times: ndarray, shape (n_samples,), optional
            Timestamps for input data. If None,
            timestamps are linearly increased for
            each sample. 
        
        Returns    
        -------
        y: ndarray, shape (n_samples,)
            Outlier scores for provided input data.
        """
        X = self._process_data(X)
        times = self._process_times(X, times)
        scores = np.empty(X.shape[0], dtype=self.params['float_type'])
        self.model.fit_predict(X, scores, times)
        return scores
        
    def window_size(self):
        """Return the number of samples in the sliding window."""
        return self.model.window_size()
        
    def get_window(self):
        """
        Return samples in the current window.
        
        Returns
        -------
        data: ndarray, shape (n_samples, n_features)
            Samples in the current window.
            
        times: ndarray, shape (n_samples,)
            Expiry times of samples in the current window.
        """
        if self.dimension == -1:
            return np.zeros([0], dtype=self.params['float_type']), np.zeros([0], dtype=self.params['float_type'])
        window_size = self.model.window_size()
        data = np.empty([window_size, self.dimension], dtype=self.params['float_type'])
        times = np.empty(window_size, dtype=self.params['float_type'])
        self.model.get_window(data, times)
        return data, times


class RSHash(OutlierDetector):
    """
    RS-Hash :cite:p:`Sathe2016`.
    
    This outlier detector assumes that features are normalized
    to a [0,1] range.

    Parameters
    ----------
    n_estimators: int
        Number of estimators in the ensemble.

    window: float
        Window length after which samples will be pruned.

    cms_w: int
        Number of hash functions per estimator for the 
        count-min sketch.

    cms_d: int
        Number of bins for the count-min sketch.

    s_param: int, optional
        The s parameter of RS-Hash, which should be an estimate
        of the number of samples in a sliding window.
        If None, the value of window will be used for s_param,
        assuming that samples arrive with an inter-arrival
        time of 1.

    float_type: np.float32 or np.float64
        The floating point type to use for internal processing.
        
    seed: int
        Random seed to use.

    n_jobs: int
        Number of threads to use for processing trees.
        Pass -1 to use as many jobs as there are CPU cores.
    """
    
    def __init__(self, n_estimators, window, cms_w, cms_d, s_param=None, float_type=np.float64, seed=0, n_jobs=-1):
        self.params = { k: v for k, v in locals().items() if k != 'self' }
        self._init_model(self.params)

    def _init_model(self, p):
        assert p['float_type'] in [np.float32, np.float64]
        assert p['n_estimators'] > 0
        assert p['window'] > 0
        cpp_obj = {np.float32: dSalmon_cpp.RSHash32, np.float64: dSalmon_cpp.RSHash64}[p['float_type']]
        self.model = cpp_obj(p['n_estimators'], p['window'], p['cms_w'],
                             p['cms_d'], p['s_param'] or p['window'], p['seed'],
                             mp.cpu_count() if p['n_jobs']==-1 else p['n_jobs'])
        self.last_time = 0
        self.dimension = -1
        
    def fit_predict(self, X, times=None):
        """
        Process next chunk of data.
        Data in X is assumed to be normalized to [0,1].

        Parameters
        ----------
        X: ndarray, shape (n_samples, n_features)
            The input data.
            
        times: ndarray, shape (n_samples,), optional
            Timestamps for input data. If None,
            timestamps are linearly increased for
            each sample. 
        
        Returns    
        -------
        y: ndarray, shape (n_samples,)
            Outlier scores for provided input data.
        """
        X = self._process_data(X)
        times = self._process_times(X, times)
        scores = np.empty(X.shape[0], dtype=self.params['float_type'])
        self.model.fit_predict(X, scores, times)
        return scores
        
    def window_size(self):
        """Return the number of samples in the sliding window."""
        return self.model.window_size()
        
    def get_window(self):
        """
        Return samples in the current window.
        
        Returns
        -------
        data: ndarray, shape (n_samples, n_features)
            Samples in the current window.
            
        times: ndarray, shape (n_samples,)
            Expiry times of samples in the current window.
        """
        if self.dimension == -1:
            return np.zeros([0], dtype=self.params['float_type']), np.zeros([0], dtype=self.params['float_type'])
        window_size = self.model.window_size()
        data = np.empty([window_size, self.dimension], dtype=self.params['float_type'])
        times = np.empty(window_size, dtype=self.params['float_type'])
        self.model.get_window(data, times)
        return data, times


class LODA(OutlierDetector):
    """
    LODA :cite:p:`Pevny2016`.
    
    This detector performs outlier detection based on equi-depth histograms.
    If random projections are used, this corresponds to the LODA algorithm,
    otherwise behaviour corresponds to a sliding window adaptation of the
    HBOS :cite:p:`Goldstein2012` algorithm.

    Parameters
    ----------
    window: float
        Window length after which samples will be pruned.

    n_projections: int, optional
        The number of random projections to use. If None,
        random projections are skipped.

    n_bins: int
        The number of histogram bins.

    float_type: np.float32 or np.float64
        The floating point type to use for internal processing.

    seed: int
        Seed for random projections.

    n_jobs: int
        Number of threads to use for processing trees.
        Pass -1 to use as many jobs as there are CPU cores.
    """

    def __init__(self, window, n_projections=None, n_bins=10, float_type=np.float64, seed=0, n_jobs=-1):
        self.params = { k: v for k, v in locals().items() if k != 'self' }
        self._init_model(self.params)

    def _perform_projections(self, X):
        if self.projector is not None:
            return self.projector.transform(X)
        return X

    def _init_model(self, p):
        assert p['float_type'] in [np.float32, np.float64]
        assert p['n_bins'] > 0
        assert p['window'] > 0
        assert p['n_projections'] is None or p['n_projections'] > 0
        cpp_obj = {np.float32: dSalmon_cpp.SWHBOS32, np.float64: dSalmon_cpp.SWHBOS64}[p['float_type']]
        self.model = cpp_obj(p['window'], p['n_bins'], mp.cpu_count() if p['n_jobs']==-1 else p['n_jobs'])
        if p['n_projections'] is not None:
            self.projector = projection.LODAProjector(p['n_projections'], float_type=p['float_type'], seed=p['seed'])
        else:
            self.projector = None
        self.last_time = 0
        self.dimension = -1

    def fit(self, X, times=None):
        """
        Process next chunk of data without returning outlier scores.
        
        Parameters
        ----------
        X: ndarray, shape (n_samples, n_features)
            The input data.
            
        times: ndarray, shape (n_samples,), optional
            Timestamps for input data. If None,
            timestamps are linearly increased for
            each sample. 
        """
        X = self._perform_projections(self._process_data(X))
        times = self._process_times(X, times)
        self.model.fit(X, times)

    def fit_predict(self, X, times = None):
        """
        Process next chunk of data.
        
        Parameters
        ----------
        X: ndarray, shape (n_samples, n_features)
            The input data.
            
        times: ndarray, shape (n_samples,), optional
            Timestamps for input data. If None,
            timestamps are linearly increased for
            each sample. 
        
        Returns    
        -------
        y: ndarray, shape (n_samples,)
            Outlier scores for provided input data.
        """
        X = self._perform_projections(self._process_data(X))
        times = self._process_times(X, times)
        scores = np.empty(X.shape[0], dtype=self.params['float_type'])
        self.model.fit_predict(X, scores, times)
        return scores
        
    def window_size(self):
        """Return the number of samples in the sliding window."""
        return self.model.window_size()
        
    def get_window(self):
        """
        Return samples in the current window.
        
        Returns
        -------
        data: ndarray, shape (n_samples, n_features)
            Samples in the current window. If n_projections is set, returns
            the projected data samples.
            
        times: ndarray, shape (n_samples,)
            Expiry times of samples in the current window.
        """
        if self.dimension == -1:
            return np.zeros([0], dtype=self.params['float_type']), np.zeros([0], dtype=self.params['float_type'])
        window_size = self.model.window_size()
        data = np.empty([window_size, self.dimension], dtype=self.params['float_type'])
        times = np.empty(window_size, dtype=self.params['float_type'])
        self.model.get_window(data, times)
        return data, times


class HSTrees(OutlierDetector):
    """
    Streaming Half-Space Trees :cite:p:`Tan2011`.

    Parameters
    ----------
    window: float
        Window length after which samples will be pruned.

    n_estimators: int
        The number of trees in the ensemble.

    max_depth: int
        The depth of each individual tree.

    size_limit: int, optional
        The maximum size of nodes to consider for outlier scoring. If None,
        defaults to 0.1*window, as described in the corresponding paper.

    float_type: np.float32 or np.float64
        The floating point type to use for internal processing.

    seed: int
        Random seed for tree construction.

    n_jobs: int
        Number of threads to use for processing trees.
        Pass -1 to use as many jobs as there are CPU cores.
    """

    # TODO: size_limit=None is inconsistent when passing times to fit_predict()
    def __init__(self, window, n_estimators, max_depth, size_limit=None, float_type=np.float64, seed=0, n_jobs=-1):
        self.params = { k: v for k, v in locals().items() if k != 'self' }
        self._init_model(self.params)
    
    def _init_model(self, p):
        assert p['float_type'] in [np.float32, np.float64]
        assert p['n_estimators'] > 0
        assert p['max_depth'] > 0
        assert p['size_limit'] is None or p['size_limit'] >= 0
        cpp_obj = {np.float32: dSalmon_cpp.HSTrees32, np.float64: dSalmon_cpp.HSTrees64}[p['float_type']]
        self.model = cpp_obj(p['window'], p['n_estimators'], p['max_depth'], p['window']//10 if p['size_limit'] is None else p['size_limit'], p['seed'], mp.cpu_count() if p['n_jobs']==-1 else p['n_jobs'])
        self.last_time = 0
        self.dimension = -1
        
    def fit_predict(self, X, times = None):
        """
        Process next chunk of data.
        
        Parameters
        ----------
        X: ndarray, shape (n_samples, n_features)
            The input data.
            
        times: ndarray, shape (n_samples,), optional
            Timestamps for input data. If None,
            timestamps are linearly increased for
            each sample. 
        
        Returns    
        -------
        y: ndarray, shape (n_samples,)
            Outlier scores for provided input data.
        """
        X = self._process_data(X)
        times = self._process_times(X, times)
        scores = np.empty(X.shape[0], dtype=self.params['float_type'])
        self.model.fit_predict(X, scores, times)
        return scores


class xStream(OutlierDetector):
    """
    xStream :cite:p:`Manzoor2018`.

    Parameters
    ----------
    window: int
        Window length after which the current window will be switch to
        the reference window.

    n_estimators: int
        The number of chains in the ensemble.

    n_projections: int
        The number of StreamHash projections to use.

    depth: int
        The length of each half-space chain.

    cms_w: int
        Number of hash functions for the count-min sketches.

    cms_d: int
        Number of bins for the count-min sketches.

    float_type: np.float32 or np.float64
        The floating point type to use for internal processing.

    seed: int
        Random seed for tree construction.

    n_jobs: int
        Number of threads to use for processing trees.
        Pass -1 to use as many jobs as there are CPU cores.
    """

    def __init__(self, window, n_estimators, n_projections, depth, cms_w=5, cms_d=1000, float_type=np.float64, seed=0, n_jobs=-1):
        self.params = { k: v for k, v in locals().items() if k != 'self' }
        self._init_model(self.params)
    
    def _init_model(self, p):
        assert p['float_type'] in [np.float32, np.float64]
        assert p['window'] > 0
        assert p['depth'] > 0
        cpp_obj = {np.float32: dSalmon_cpp.HSChains32, np.float64: dSalmon_cpp.HSChains64}[p['float_type']]
        self.model = cpp_obj(p['window'], p['n_estimators'], p['depth'], p['cms_w'], p['cms_d'], p['seed'], mp.cpu_count() if p['n_jobs']==-1 else p['n_jobs'])
        self.projector = projection.StreamHash(n_projections, float_type=p['float_type'], seed=p['seed'])
        self.initial_sample = np.empty((0, p['n_projections']), dtype=p['float_type'])
        self.initial_sample_was_set = False

    def set_initial_sample(self, data, features=None):
        """
        Optionally set the initial sample used for estimating the range of
        projected features. If no initial sample is provided, ranges will be
        estimated from the first `window` data points. In this case, the first
        `window` data points are stored to construct the reference window as
        soon as range estimates are available.

        Parameters
        ----------
        data: ndarray, shape (n_samples, n_features)
            The initial sample.
            
        features: list, optional
            Feature names used for StreamHash. The `repr()` of list elements
            is used as basis for hashing, hence elements do not necessarily 
            have to be strings. If None, `range(n_features)` is used as
            feature names.
        """
        assert not self.initial_sample_was_set and not self.initial_sample.size, 'The initial sample must be set before processing any data point.'
        data = sanitizeData(data, self.params['float_type'])
        data_projected = self.projector.transform(data, features)
        self.model.set_initial_minmax(np.min(data_projected, axis=0), np.max(data_projected, axis=0))
        self.initial_sample_was_set = True
        self.initial_sample = None
        
    def fit_predict(self, X, features=None):
        """
        Process next chunk of data.
        
        Parameters
        ----------
        X: ndarray, shape (n_samples, n_features)
            The input data.
            
        features: list, optional
            Feature names used for StreamHash. The `repr()` of list elements
            is used as basis for hashing, hence elements do not necessarily 
            have to be strings. If None, `range(n_features)` is used as
            feature names.
        
        Returns    
        -------
        y: ndarray, shape (n_samples,)
            Outlier scores for provided input data.
        """
        X = sanitizeData(X, self.params['float_type'])
        X_projected = self.projector.transform(X, features)
        returned_scores = np.empty(X.shape[0], dtype=self.params['float_type'])
        if not self.initial_sample_was_set:
            window = self.params['window']
            threshold = window - len(self.initial_sample)
            self.initial_sample = np.append(self.initial_sample, X_projected[:threshold], axis=0)
            returned_scores[:threshold] = np.NaN
            if len(self.initial_sample) < window:
                return returned_scores
            self.initial_sample_was_set = True
            self.model.set_initial_minmax(np.min(self.initial_sample, axis=0), np.max(self.initial_sample, axis=0))
            self.model.fit(self.initial_sample)
            self.initial_sample = None
            X_projected = X_projected[threshold:]
            scores = returned_scores[threshold:]
        else:
            scores = returned_scores
        self.model.fit_predict(X_projected, scores)
        return returned_scores
