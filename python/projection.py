# Copyright (c) 2020 CN Group, TU Wien
# Released under the GNU Lesser General Public License version 3,
# see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

"""
Feature projectors.
"""

import numpy as np
import struct
import random
import zlib

class LODAProjector(object):
    """
    Sparse random projections as used for by LODA :cite:p:`Pevny2016`.

    Parameters
    ----------
    n_projections: int
        The dimension of the projected data.

    float_type: np.float32 or np.float64
        The floating point type to use for internal processing.

    seed: int
        Random seed for projection.
    """
    def __init__(self, n_projections, float_type=np.float64, seed=0):
        self.n_projections = n_projections
        self.float_type = float_type
        self.seed = seed
        self.proj_matrix = None

    def _init_projections(self, dimension):
        rng = random.Random(self.seed)
        nprng = np.random.RandomState(self.seed)
        self.proj_matrix = np.zeros((dimension,self.n_projections), dtype=self.float_type)
        proj_per_histogram = int(round(np.sqrt(dimension)))
        for i in range(self.n_projections):
            indices = rng.sample(range(dimension), k=proj_per_histogram)
            self.proj_matrix[indices,i] = nprng.normal(size=proj_per_histogram)

    def transform(self, X):
        """
        Perform projection of a block of data. Order of rows in `X` is not
        important.

        Parameters
        ----------
        X: ndarray, shape (n_samples, n_features)
            The input data.

        Returns
        -------
        X_tr: ndarray, shape (n_samples, n_features)
            The projected data.
        """
        if self.proj_matrix is None:
            self._init_projections(X.shape[1])
        return np.matmul(X, self.proj_matrix)

class StreamHash(object):
    """
    Random projections for feature-evolving streams as used by
    xStream :cite:p:`Manzoor2018`.

    Parameters
    ----------
    n_projections: int
        The dimension of the projected data.

    float_type: np.float32 or np.float64
        The floating point type to use for internal processing.

    seed: int
        Random seed for projection.
    """
    def __init__(self, n_projections, float_type=np.float64, seed=0):
        self.n_projections = n_projections
        self.float_type = float_type
        self.seed = seed

    def _streamhash_vec(self, feature):
        hash_base = zlib.crc32(struct.pack('<Is', self.seed, repr(feature).encode()))
        h = [ zlib.crc32(struct.pack('<I',i), hash_base) % 6 for i in range(self.n_projections) ]
        h = np.array(h, dtype=self.float_type)
        mask1 = h < 1
        mask2 = h >= 5
        h[mask1] = -1
        h[~mask1 & ~mask2] = 0
        h[mask2] = 1
        return (3/self.n_projections)**0.5 * h

    def transform(self, X, features=None):
        """
        Perform projection of a block of data. Order of rows in `X` is not
        important.

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
        X_tr: ndarray, shape (n_samples, n_features)
            The projected data.
        """
        if features is None:
            features = range(X.shape[1])
        projection_matrix = np.stack([ self._streamhash_vec(feature) for feature in features ])
        return np.matmul(X, projection_matrix)
        