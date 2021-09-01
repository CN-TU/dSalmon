# Copyright (c) 2020 CN Group, TU Wien
# Released under the GNU Lesser General Public License version 3,
# see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

import numpy as np
import random
import multiprocessing as mp

from dSalmon import swig as dSalmon_cpp
from dSalmon.util import sanitizeData, sanitizeTimes, lookupDistance

class OutlierDetector(object):
	def _init_model(self, p):
		pass

	def get_params(self, deep=True):
		return self.params

	def set_params(self, **params):
		p = self.params.copy()
		for key in params:
			assert key in p, 'Unknown parameter: %s' % key
			p[key] = params[key]
		self._init_model(p)

	def _processData(self, data):
		data = sanitizeData(data, self.params['float_type'])
		assert self.dimension == -1 or data.shape[1] == self.dimension
		self.dimension = data.shape[1]
		return data

	def _processTimes(self, data, times):
		times = sanitizeTimes(times, data.shape[0], self.last_time, self.params['float_type'])
		# if times is None:
		# 	times = np.arange(self.last_time + 1, self.last_time + 1 + data.shape[0])
		# else:
		# 	times = np.array(times, dtype=self.params['float_type'])
		# 	assert len(times.shape) <= 1
		# 	if len(times.shape) == 0:
		# 		times = np.repeat(times[None], data.shape[0])
		# 	else:
		# 		assert times.shape[0] == data.shape[0]
		self.last_time = times[-1]
		return times


class SWDBOR(OutlierDetector):
	"""Distance based outlier detection by radius."""
	
	def __init__(self, window, radius, metric='euclidean', float_type=np.float64, min_node_size=5, max_node_size=100, split_sampling=20):
		"""
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
		self.params = { k: v for k, v in locals().items() if k != 'self' }
		self._init_model(self.params)

	def _init_model(self, p):
		assert p['float_type'] in [np.float32, np.float64]
		assert p['max_node_size'] > 2 * p['min_node_size'], 'max_node_size must be > 2 * min_node_size'
		assert p['min_node_size'] > 0
		assert p['window'] > 0
		assert p['radius'] > 0
		distance_function = lookupDistance(p['metric'], p['float_type'])
		cpp_obj = {np.float32: dSalmon_cpp.DBOR32, np.float64: dSalmon_cpp.DBOR64}[p['float_type']]
		self.model = cpp_obj(p['window'], p['radius'], distance_function, p['min_node_size'],
							 p['max_node_size'], p['split_sampling'])
		self.last_time = 0
		self.dimension = -1
		self.params = p

	def fit_predict(self, data, times=None):
		"""
		Process next chunk of data.
		
		Parameters
		---------------
		X: ndarray, shape (n_samples, n_features)
			The input data.
			
		times: ndarray, shape (n_samples,), optional
			Timestamps for input data. If None,
			timestamps are linearly increased for
			each sample. 
		
		Returns	
		---------------
		y: ndarray, shape (n_samples,)
			Outlier scores for provided input data.
		"""
		data = self._processData(data)
		times = self._processTimes(data, times)
		scores = np.empty(data.shape[0], dtype=self.params['float_type'])
		self.model.fit_predict(data, scores, np.array(times, dtype=self.params['float_type']))
		return scores
		
	def window_size(self):
		"""Return the number of samples in the sliding window."""
		return self.model.window_size()
		
	def get_window(self):
		"""
		Return samples in the current window.
		
		Returns
		---------------
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
	"""Distance based outlier detection by k nearest neighbors."""
	
	def __init__(self, window, k, k_is_max=False, metric='euclidean', float_type=np.float64, min_node_size = 5, max_node_size = 100, split_sampling = 20):
		"""
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

		metric: string
			Which distance metric to use. Currently supported metrics
			include 'chebyshev', 'cityblock', 'euclidean' and
			'minkowsi'.

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
		self.params = { k: v for k, v in locals().items() if k != 'self' }
		self._init_model(self.params)

	def _init_model(self, p):
		assert p['float_type'] in [np.float32, np.float64]
		assert p['max_node_size'] > 2 * p['min_node_size'], 'max_node_size must be > 2 * min_node_size'
		assert p['min_node_size'] > 0
		assert p['window'] > 0
		assert p['k'] > 0
		distance_function = lookupDistance(p['metric'], p['float_type'])
		cpp_obj = {np.float32: dSalmon_cpp.SWKNN32, np.float64: dSalmon_cpp.SWKNN64}[p['float_type']]
		self.model = cpp_obj(p['window'], p['k'], distance_function, p['min_node_size'],
							 p['max_node_size'], p['split_sampling'])
		self.last_time = 0
		self.dimension = -1
		
	def fit_predict(self, data, times=None):
		"""
		Process next chunk of data.
		
		Parameters
		---------------
		X: ndarray, shape (n_samples, n_features)
			The input data.
			
		times: ndarray, shape (n_samples,), optional
			Timestamps for input data. If None,
			timestamps are linearly increased for
			each sample. 
		
		Returns	
		---------------
		y: ndarray, shape (n_samples,) or (n_samples,k)
			Outlier scores for provided input data.
		"""
		data = self._processData(data)
		times = self._processTimes(data, times)
		if self.params['k_is_max']:
			scores = np.empty([data.shape[0], self.params['k']], dtype=self.params['float_type'])
			self.model.fit_predict_with_neighbors(data, scores, np.array(times, dtype=self.params['float_type']))
		else:
			scores = np.empty(data.shape[0], dtype=self.params['float_type'])
			self.model.fit_predict(data, scores, np.array(times, dtype=self.params['float_type']))
		return scores

	def window_size(self):
		"""Return the number of samples in the sliding window."""
		return self.model.window_size()
		
	def get_window(self):
		"""
		Return samples in the current window.
		
		Returns
		---------------
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
	"""Sliding Window Local Outlier Factor."""
	
	def __init__(self, window, k, simplified=False, k_is_max=False, metric='euclidean', float_type=np.float64, min_node_size=5, max_node_size=100, split_sampling=20):
		"""
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

		metric: string
			Which distance metric to use. Currently supported metrics
			include 'chebyshev', 'cityblock', 'euclidean' and
			'minkowsi'.

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
		self.params = { k: v for k, v in locals().items() if k != 'self' }
		self._init_model(self.params)

	def _init_model(self, p):
		assert p['float_type'] in [np.float32, np.float64]
		assert p['max_node_size'] > 2 * p['min_node_size'], 'max_node_size must be > 2 * min_node_size'
		assert p['min_node_size'] > 0
		assert p['window'] > 0
		assert p['k'] > 0
		distance_function = lookupDistance(p['metric'], p['float_type'])
		cpp_obj = {np.float32: dSalmon_cpp.SWLOF32, np.float64: dSalmon_cpp.SWLOF64}[p['float_type']]
		self.model = cpp_obj(p['window'], p['k'], p['simplified'], distance_function,
							 p['min_node_size'], p['max_node_size'], p['split_sampling'])
		self.last_time = 0
		self.dimension = -1
		
	def fit(self, data, times=None):
		"""
		Process next chunk of data without returning scores.
		
		Parameters
		---------------
		X: ndarray, shape (n_samples, n_features)
			The input data.
			
		times: ndarray, shape (n_samples,), optional
			Timestamps for input data. If None,
			timestamps are linearly increased for
			each sample. 
		"""
		data = self._processData(data)
		times = self._processTimes(data, times)
		self.model.fit(data, np.array(times, dtype=self.params['float_type']))
		
	def fit_predict(self, data, times=None):
		"""
		Process next chunk of data.
		
		Parameters
		---------------
		X: ndarray, shape (n_samples, n_features)
			The input data.
			
		times: ndarray, shape (n_samples,), optional
			Timestamps for input data. If None,
			timestamps are linearly increased for
			each sample. 
		
		Returns	
		---------------
		y: ndarray, shape (n_samples,) or (n_samples,k)
			Outlier scores for provided input data.
		"""
		data = self._processData(data)
		times = self._processTimes(data, times)
		scores = np.empty([data.shape[0], self.params['k']], dtype=self.params['float_type'])
		self.model.fit_predict(data, scores, np.array(times, dtype=self.params['float_type']))
		return scores if self.params['k_is_max'] else scores[:,-1]
		
	def window_size(self):
		"""Return the number of samples in the sliding window."""
		return self.model.window_size()
		
	def get_window(self):
		"""
		Return samples in the current window.
		
		Returns
		---------------
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
	"""Streaming outlier detection based on Sparse Data Observers."""

	def __init__(self, k, T, qv=0.3, x=6, freq_bins=10, max_freq = 6.283, metric='euclidean', float_type=np.float64, seed=0, return_sampling=False):
		"""
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

		float_type: np.float32 or np.float64
			The floating point type to use for internal processing.

		seed: int (default=0)
			Random seed to use.
			
		return_sampling: bool (default=False)
			Also return whether a data point was adopted as observer.
		"""
		self.params = { k: v for k, v in locals().items() if k != 'self' }
		self._init_model(self.params)

	def _init_model(self, p):
		assert p['float_type'] in [np.float32, np.float64]
		assert 0 <= p['qv'] < 1, 'qv must be in [0,1)'
		assert p['x'] > 0, 'x must be > 0'
		assert p['k'] > 0, 'k must be > 0'
		assert p['T'] > 0, 'T must be > 0'
		distance_function = lookupDistance(p['metric'], p['float_type'])
		cpp_obj = {np.float32: dSalmon_cpp.SDOstream32, np.float64: dSalmon_cpp.SDOstream64}[p['float_type']]
		self.model = cpp_obj(p['k'], p['T'], p['qv'], p['x'], p['freq_bins'], p['max_freq'], distance_function, p['seed'])
		self.last_time = 0
		self.dimension = -1
		
	def fit_predict(self, data, times=None):
		"""
		Process next chunk of data.
		
		Parameters
		---------------
		X: ndarray, shape (n_samples, n_features)
			The input data.
			
		times: ndarray, shape (n_samples,), optional
			Timestamps for input data. If None,
			timestamps are linearly increased for
			each sample. 
		
		Returns	
		---------------
		y: ndarray, shape (n_samples,)
			Outlier scores for provided input data.
		"""
		data = self._processData(data)
		times = self._processTimes(data, times)
		scores = np.empty(data.shape[0], dtype=self.params['float_type'])
		if self.params['return_sampling']:
			sampling = np.empty(data.shape[0], dtype=np.int32)
			self.model.fit_predict_with_sampling(data, scores, np.array(times, dtype=self.params['float_type']), sampling)
			return scores, sampling
		else:
			self.model.fit_predict(data, scores, np.array(times, dtype=self.params['float_type']))
			return scores
		
	def observer_count(self):
		"""Return the current number of observers."""
		return self.model.observer_count()
		
	def get_observers(self, time=None):
		"""
		Return observer data.
		
		Returns	
		---------------
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
		freq_bins = self.model.frequency_bin_count()
		complex_type = np.complex64 if self.params['float_type'] == np.float32 else np.complex128
		if observer_cnt == 0:
			return np.zeros([0], dtype=self.params['float_type']), np.zeros([0,freq_bins], dtype=complex_type), np.zeros([0], dtype=self.params['float_type'])
		data = np.empty([observer_cnt, self.dimension], dtype=self.params['float_type'])
		observations = np.empty([observer_cnt, freq_bins], dtype=complex_type)
		av_observations = np.empty(observer_cnt, dtype=self.params['float_type'])
		self.model.get_observers(data, observations, av_observations, self.params['float_type'](time))
		return data, observations, av_observations


class SWRRCT(OutlierDetector):
	"""Robust Random Cut Forest."""
	
	def __init__(self, window, n_estimators = 10, float_type=np.float64, seed=0, n_jobs=-1):
		"""
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
		
	def fit_predict(self, data, times=None):
		"""
		Process next chunk of data.
		
		Parameters
		---------------
		X: ndarray, shape (n_samples, n_features)
			The input data.
			
		times: ndarray, shape (n_samples,), optional
			Timestamps for input data. If None,
			timestamps are linearly increased for
			each sample. 
		
		Returns	
		---------------
		y: ndarray, shape (n_samples,)
			Outlier scores for provided input data.
		"""
		data = self._processData(data)
		times = self._processTimes(data, times)
		scores = np.empty(data.shape[0], dtype=self.params['float_type'])
		self.model.fit_predict(data, scores, np.array(times, dtype=self.params['float_type']))
		return scores
		
	def window_size(self):
		"""Return the number of samples in the sliding window."""
		return self.model.window_size()
		
	def get_window(self):
		"""
		Return samples in the current window.
		
		Returns
		---------------
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
	"""RS-Hash."""
	
	def __init__(self, n_estimators, window, cms_w, cms_d, s_param=None, float_type=np.float64, seed=0, n_jobs=-1):
		"""
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
			The s param of RS-Hash, which should be an estimate
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
		
	def fit_predict(self, data, times=None):
		"""
		Process next chunk of data.
		Data in X is assumed to be normalized to [0,1].

		Parameters
		---------------
		X: ndarray, shape (n_samples, n_features)
			The input data.
			
		times: ndarray, shape (n_samples,), optional
			Timestamps for input data. If None,
			timestamps are linearly increased for
			each sample. 
		
		Returns	
		---------------
		y: ndarray, shape (n_samples,)
			Outlier scores for provided input data.
		"""
		data = self._processData(data)
		times = self._processTimes(data, times)
		scores = np.empty(data.shape[0], dtype=self.params['float_type'])
		self.model.fit_predict(data, scores, np.array(times, dtype=self.params['float_type']))
		return scores
		
	def window_size(self):
		"""Return the number of samples in the sliding window."""
		return self.model.window_size()
		
	def get_window(self):
		"""
		Return samples in the current window.
		
		Returns
		---------------
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
	LODA.
	
	This detector performs outlier detection based on equi-depth histograms.
	If random projections are used, this corresponds to the LODA algorithm,
	otherwise behaviour corresponds to a sliding window adaptation of the
	HBOS algorithm.
	"""

	def __init__(self, window, n_projections=None, n_bins=10, float_type=np.float64, seed=0, n_jobs=-1):
		"""
		Parameters
		----------
		window: float
			Window length after which samples will be pruned.

		n_projections: int
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
		self.params = { k: v for k, v in locals().items() if k != 'self' }
		self._init_model(self.params)
	
	def _init_projections(self):
		rng = random.Random(self.params['seed'])
		nprng = np.random.RandomState(self.params['seed'])
		n_projections = self.params['n_projections']
		self.proj_matrix = np.zeros((self.dimension,n_projections), dtype=self.params['float_type'])
		proj_per_histogram = int(round(np.sqrt(self.dimension)))
		for i in range(n_projections):
			indices = rng.sample(range(self.dimension), k=proj_per_histogram)
			self.proj_matrix[indices,i] = nprng.normal(size=proj_per_histogram)

	def _init_model(self, p):
		assert p['float_type'] in [np.float32, np.float64]
		assert p['n_bins'] > 0
		assert p['window'] > 0
		assert p['n_projections'] is None or p['n_projections'] > 0
		cpp_obj = {np.float32: dSalmon_cpp.SWHBOS32, np.float64: dSalmon_cpp.SWHBOS64}[p['float_type']]
		self.model = cpp_obj(p['window'], p['n_bins'], mp.cpu_count() if p['n_jobs']==-1 else p['n_jobs'])
		self.last_time = 0
		self.dimension = -1
		self.proj_matrix = None
		
	def fit_predict(self, data, times = None):
		"""
		Process next chunk of data.
		
		Parameters
		---------------
		X: ndarray, shape (n_samples, n_features)
			The input data.
			
		times: ndarray, shape (n_samples,), optional
			Timestamps for input data. If None,
			timestamps are linearly increased for
			each sample. 
		
		Returns	
		---------------
		y: ndarray, shape (n_samples,)
			Outlier scores for provided input data.
		"""
		data = self._processData(data)
		if self.params['n_projections'] is not None:
			if self.proj_matrix is None:
				self._init_projections()
			data = np.matmul(data, self.proj_matrix)
		times = self._processTimes(data, times)
		scores = np.empty(data.shape[0], dtype=self.params['float_type'])
		self.model.fit_predict(data, scores, np.array(times, dtype=self.params['float_type']))
		return scores
		
	def window_size(self):
		"""Return the number of samples in the sliding window."""
		return self.model.window_size()
		
	def get_window(self):
		"""
		Return samples in the current window.
		
		Returns
		---------------
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
	Streaming Half-Space-Trees.
	"""

	# TODO: size_limit=None is inconsistent when passing times to fit_predict()
	def __init__(self, window, n_estimators, max_depth, size_limit=None, float_type=np.float64, seed=0):
		"""
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
		"""
		self.params = { k: v for k, v in locals().items() if k != 'self' }
		self._init_model(self.params)
	
	def _init_model(self, p):
		assert p['float_type'] in [np.float32, np.float64]
		assert p['n_estimators'] > 0
		assert p['max_depth'] > 0
		assert p['size_limit'] is None or p['size_limit'] > 0
		cpp_obj = {np.float32: dSalmon_cpp.HSTrees32, np.float64: dSalmon_cpp.HSTrees64}[p['float_type']]
		self.model = cpp_obj(p['window'], p['n_estimators'], p['max_depth'], p['window']//10 if p['size_limit'] is None else p['size_limit'], p['seed'])
		self.last_time = 0
		self.dimension = -1
		
	def fit_predict(self, data, times = None):
		"""
		Process next chunk of data.
		
		Parameters
		---------------
		X: ndarray, shape (n_samples, n_features)
			The input data.
			
		times: ndarray, shape (n_samples,), optional
			Timestamps for input data. If None,
			timestamps are linearly increased for
			each sample. 
		
		Returns	
		---------------
		y: ndarray, shape (n_samples,)
			Outlier scores for provided input data.
		"""
		data = self._processData(data)
		times = self._processTimes(data, times)
		scores = np.empty(data.shape[0], dtype=self.params['float_type'])
		self.model.fit_predict(data, scores, np.array(times, dtype=self.params['float_type']))
		return scores
