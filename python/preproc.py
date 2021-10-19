# Copyright (c) 2020 CN Group, TU Wien
# Released under the GNU Lesser General Public License version 3,
# see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

"""
Tools for preprocessing.
"""

import numpy as np

from dSalmon import swig as dSalmon_cpp
from dSalmon.util import sanitizeData, sanitizeTimes

class SWScaler(object):
    def __init__(self, float_type=np.float64):
        self.float_type = float_type
        self.scaler = None
        self.last_time = 0
        self.dimension = -1

    def transform(self, data, times=None):
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
        X_tr: ndarray, shape (n_samples, n_features)
            Transformed input data.
        """
        X = sanitizeData(X, self.float_type)
        assert self.dimension == -1 or X.shape[1] == self.dimension
        self.dimension = X.shape[1]
        times = sanitizeTimes(times, X.shape[0], self.last_time, self.float_type)
        self.last_time = times[-1]
        X_tr = np.empty_like(X)
        self.scaler.transform(X, X_tr, times)
        return X_tr

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
        cpp_obj = {np.float32: dSalmon_cpp.SWZScoreScaler32, np.float64: dSalmon_cpp.SWZScoreScaler64}[float_type]
        self.scaler = cpp_obj(window)

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
        cpp_obj = {np.float32: dSalmon_cpp.SWQuantileScaler32, np.float64: dSalmon_cpp.SWQuantileScaler64}[float_type]
        self.scaler = cpp_obj(window, quantile)
