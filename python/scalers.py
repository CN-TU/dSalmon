# Copyright (c) 2020 CN Group, TU Wien
# Released under the GNU Lesser General Public License version 3,
# see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

"""
Scalers for streaming data.
"""

import numpy as np

from dSalmon import swig as dSalmon_cpp
from dSalmon.util import sanitizeData, sanitizeTimes

class SWScaler(object):
    """
    Base class for sliding window scalers.
    """
    
    def __init__(self, float_type=np.float64):
        self.float_type = float_type
        self.last_time = 0
        self.dimension = -1

    def _transform(self, X, times):
        raise NotImplementedError()

    def transform(self, X, times=None):
        """
        Transform the next chunk of data.
        
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
        X_tr: ndarray, shape (n_samples, n_features)
            Transformed input data.
        """
        X_tr = sanitizeData(X, self.float_type)
        if X_tr is X:
            X_tr = X.copy()
        assert self.dimension == -1 or X_tr.shape[1] == self.dimension
        self.dimension = X_tr.shape[1]
        times = sanitizeTimes(times, X_tr.shape[0], self.last_time, self.float_type)
        self.last_time = times[-1]
        self._transform(X_tr, times)
        return X_tr

    def transform_inplace(self, X, times=None):
        """
        Transform the next chunk of data in-place. Requires
        `X` to be a C-style contiguous `ndarray`.
        
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
        X_tr: ndarray, shape (n_samples, n_features)
            Transformed input data. Equal to `X`.
        """
        assert isinstance(X, np.ndarray) and len(X.shape) in (0,1,2)
        assert X.flags['C_CONTIGUOUS'] and X.flags['WRITEABLE']
        if len(X.shape) < 2:
            X = X.reshape((1,-1))
        assert self.dimension == -1 or X.shape[1] == self.dimension
        self.dimension = X.shape[1]
        times = sanitizeTimes(times, X.shape[0], self.last_time, self.float_type)
        self.last_time = times[-1]
        self._transform(X, times)
        return X

class SWZScoreScaler(SWScaler):
    """
    Performs z-score normalization of samples based on mean and standard
    deviation observed in a sliding window of length `window`.

    Parameters
    ----------
    window: float
        Window length after which samples will be pruned.

    float_type: np.float32 or np.float64
        The floating point type to use for internal processing.
    """

    def __init__(self, window, float_type=np.float64):
        super().__init__(float_type)
        self.window = window
        cpp_obj = {np.float32: dSalmon_cpp.StatisticsTree32, np.float64: dSalmon_cpp.StatisticsTree64}[float_type]
        self.tree = cpp_obj(window)

    def _transform(self, X, times):
        self.tree.transform_zscore(X, times)

class SWQuantileScaler(SWScaler):
    """
    Performs normalization so that the p-quantile of the current sliding
    window is mapped to 0 and the (1-p)-quantile is mapped to 1. If
    `quantile==0`, performs minmax normalization. Note that due to its
    lacking robustness, minmax normalization is likely to result in unstable
    results for stream data.
    
    Parameters
    ----------
    window: float
        Window length after which samples will be pruned.

    quantile: float with 0 <= quantile < 0.5
        The quantile value for computing reference values.

    float_type: np.float32 or np.float64
        The floating point type to use for internal processing.
    """

    def __init__(self, window, quantile, float_type=np.float64):
        super().__init__(float_type)
        self.window = window
        self.quantile = quantile
        cpp_obj = {np.float32: dSalmon_cpp.StatisticsTree32, np.float64: dSalmon_cpp.StatisticsTree64}[float_type]
        self.tree = cpp_obj(window)

    def _transform(self, X, times):
        self.tree.transform_quantile(X, times, self.quantile)