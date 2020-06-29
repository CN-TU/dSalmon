# Copyright (c) 2020 CN Group, TU Wien
# Released under the GNU Lesser General Public License version 3,
# see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

import numpy as np
from . import swig

def sanitizeData(data, float_type=np.float64):
	data = np.array(data, dtype=float_type)
	if len(data.shape) == 1:
		data = data[None,:]
	assert len(data.shape) == 2
	assert not np.isnan(data).any(), 'NaN values are not allowed'
	return data

def lookupDistance(name, float_type, **kwargs):
	wrappers = {
		'chebyshev': 'ChebyshevDist',
		'cityblock': 'ManhattanDist',
		'euclidean': 'EuclideanDist',
	}
	suffix = {np.float32: '32', np.float64: '64'}[float_type]

	if name in wrappers:
		return swig.__dict__[wrappers[name] + suffix]()
	elif name == 'minkowski':
		if not 'p' in kwargs:
			raise TypeError('p is required for minkowski distance')
		return swig.__dict__['MinkowsiDist' + suffix](kwargs['p'])
	else:
		raise TypeError('Unknown metric')
