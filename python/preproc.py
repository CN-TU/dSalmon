# Copyright (c) 2020 CN Group, TU Wien
# Released under the GNU Lesser General Public License version 3,
# see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

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
        data = sanitizeData(data, self.float_type)
        assert self.dimension == -1 or data.shape[1] == self.dimension
        self.dimension = data.shape[1]
        times = sanitizeTimes(times, data.shape[0], self.last_time, self.float_type)
        self.last_time = times[-1]
        data_normalized = np.empty_like(data)
        self.scaler.transform(data, data_normalized, times)
        return data_normalized

class SWZScoreScaler(SWScaler):
    def __init__(self, window, float_type=np.float64):
        super().__init__(float_type)
        self.window = window
        cpp_obj = {np.float32: dSalmon_cpp.SWZScoreScaler32, np.float64: dSalmon_cpp.SWZScoreScaler64}[float_type]
        self.scaler = cpp_obj(window)

class SWQuantileScaler(SWScaler):
    def __init__(self, window, quantile, float_type=np.float64):
        super().__init__(float_type)
        self.window = window
        self.quantile = quantile
        cpp_obj = {np.float32: dSalmon_cpp.SWQuantileScaler32, np.float64: dSalmon_cpp.SWQuantileScaler64}[float_type]
        self.scaler = cpp_obj(window, quantile)
