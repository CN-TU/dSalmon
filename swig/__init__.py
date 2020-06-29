# This file was automatically generated by SWIG (http://www.swig.org).
# Version 4.0.2
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
if _swig_python_version_info < (2, 7, 0):
    raise RuntimeError("Python 2.7 or later required")

# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _dSalmon
else:
    import _dSalmon

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)


def _swig_setattr_nondynamic_instance_variable(set):
    def set_instance_attr(self, name, value):
        if name == "thisown":
            self.this.own(value)
        elif name == "this":
            set(self, name, value)
        elif hasattr(self, name) and isinstance(getattr(type(self), name), property):
            set(self, name, value)
        else:
            raise AttributeError("You cannot add instance attributes to %s" % self)
    return set_instance_attr


def _swig_setattr_nondynamic_class_variable(set):
    def set_class_attr(cls, name, value):
        if hasattr(cls, name) and not isinstance(getattr(cls, name), property):
            set(cls, name, value)
        else:
            raise AttributeError("You cannot add class attributes to %s" % cls)
    return set_class_attr


def _swig_add_metaclass(metaclass):
    """Class decorator for adding a metaclass to a SWIG wrapped class - a slimmed down version of six.add_metaclass"""
    def wrapper(cls):
        return metaclass(cls.__name__, cls.__bases__, cls.__dict__.copy())
    return wrapper


class _SwigNonDynamicMeta(type):
    """Meta class to enforce nondynamic attributes (no new attributes) for a class"""
    __setattr__ = _swig_setattr_nondynamic_class_variable(type.__setattr__)


class Distance32(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr

# Register Distance32 in _dSalmon:
_dSalmon.Distance32_swigregister(Distance32)

class Distance64(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr

# Register Distance64 in _dSalmon:
_dSalmon.Distance64_swigregister(Distance64)

class EuclideanDist32(Distance32):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self):
        _dSalmon.EuclideanDist32_swiginit(self, _dSalmon.new_EuclideanDist32())
    __swig_destroy__ = _dSalmon.delete_EuclideanDist32

# Register EuclideanDist32 in _dSalmon:
_dSalmon.EuclideanDist32_swigregister(EuclideanDist32)

class EuclideanDist64(Distance64):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self):
        _dSalmon.EuclideanDist64_swiginit(self, _dSalmon.new_EuclideanDist64())
    __swig_destroy__ = _dSalmon.delete_EuclideanDist64

# Register EuclideanDist64 in _dSalmon:
_dSalmon.EuclideanDist64_swigregister(EuclideanDist64)

class ManhattanDist32(Distance32):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self):
        _dSalmon.ManhattanDist32_swiginit(self, _dSalmon.new_ManhattanDist32())
    __swig_destroy__ = _dSalmon.delete_ManhattanDist32

# Register ManhattanDist32 in _dSalmon:
_dSalmon.ManhattanDist32_swigregister(ManhattanDist32)

class ManhattanDist64(Distance64):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self):
        _dSalmon.ManhattanDist64_swiginit(self, _dSalmon.new_ManhattanDist64())
    __swig_destroy__ = _dSalmon.delete_ManhattanDist64

# Register ManhattanDist64 in _dSalmon:
_dSalmon.ManhattanDist64_swigregister(ManhattanDist64)

class ChebyshevDist32(Distance32):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self):
        _dSalmon.ChebyshevDist32_swiginit(self, _dSalmon.new_ChebyshevDist32())
    __swig_destroy__ = _dSalmon.delete_ChebyshevDist32

# Register ChebyshevDist32 in _dSalmon:
_dSalmon.ChebyshevDist32_swigregister(ChebyshevDist32)

class ChebyshevDist64(Distance64):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self):
        _dSalmon.ChebyshevDist64_swiginit(self, _dSalmon.new_ChebyshevDist64())
    __swig_destroy__ = _dSalmon.delete_ChebyshevDist64

# Register ChebyshevDist64 in _dSalmon:
_dSalmon.ChebyshevDist64_swigregister(ChebyshevDist64)

class MinkowskiDist32(Distance32):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, p):
        _dSalmon.MinkowskiDist32_swiginit(self, _dSalmon.new_MinkowskiDist32(p))
    __swig_destroy__ = _dSalmon.delete_MinkowskiDist32

# Register MinkowskiDist32 in _dSalmon:
_dSalmon.MinkowskiDist32_swigregister(MinkowskiDist32)

class MinkowskiDist64(Distance64):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, p):
        _dSalmon.MinkowskiDist64_swiginit(self, _dSalmon.new_MinkowskiDist64(p))
    __swig_destroy__ = _dSalmon.delete_MinkowskiDist64

# Register MinkowskiDist64 in _dSalmon:
_dSalmon.MinkowskiDist64_swigregister(MinkowskiDist64)

class MTree32(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, distance, min_node_size, max_node_size, split_sampling, insert_jobs, query_jobs):
        _dSalmon.MTree32_swiginit(self, _dSalmon.new_MTree32(distance, min_node_size, max_node_size, split_sampling, insert_jobs, query_jobs))

    def insert(self, data, indices):
        return _dSalmon.MTree32_insert(self, data, indices)

    def remove(self, selector):
        return _dSalmon.MTree32_remove(self, selector)

    def update(self, keys, data):
        return _dSalmon.MTree32_update(self, keys, data)

    def updateByIndex(self, keys, data):
        return _dSalmon.MTree32_updateByIndex(self, keys, data)

    def updateByIndexSlice(self, _from, to, step, data):
        return _dSalmon.MTree32_updateByIndexSlice(self, _from, to, step, data)

    def keys(self, selector, keys):
        return _dSalmon.MTree32_keys(self, selector, keys)

    def indices(self, selector, indices):
        return _dSalmon.MTree32_indices(self, selector, indices)

    def getPoints(self, selector, data):
        return _dSalmon.MTree32_getPoints(self, selector, data)

    def size(self):
        return _dSalmon.MTree32_size(self)

    def dataDimension(self):
        return _dSalmon.MTree32_dataDimension(self)

    def clear(self):
        return _dSalmon.MTree32_clear(self)

    def serialize(self):
        return _dSalmon.MTree32_serialize(self)

    def unserialize(self, data):
        return _dSalmon.MTree32_unserialize(self, data)
    __swig_destroy__ = _dSalmon.delete_MTree32

# Register MTree32 in _dSalmon:
_dSalmon.MTree32_swigregister(MTree32)

class MTree64(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, distance, min_node_size, max_node_size, split_sampling, insert_jobs, query_jobs):
        _dSalmon.MTree64_swiginit(self, _dSalmon.new_MTree64(distance, min_node_size, max_node_size, split_sampling, insert_jobs, query_jobs))

    def insert(self, data, indices):
        return _dSalmon.MTree64_insert(self, data, indices)

    def remove(self, selector):
        return _dSalmon.MTree64_remove(self, selector)

    def update(self, keys, data):
        return _dSalmon.MTree64_update(self, keys, data)

    def updateByIndex(self, keys, data):
        return _dSalmon.MTree64_updateByIndex(self, keys, data)

    def updateByIndexSlice(self, _from, to, step, data):
        return _dSalmon.MTree64_updateByIndexSlice(self, _from, to, step, data)

    def keys(self, selector, keys):
        return _dSalmon.MTree64_keys(self, selector, keys)

    def indices(self, selector, indices):
        return _dSalmon.MTree64_indices(self, selector, indices)

    def getPoints(self, selector, data):
        return _dSalmon.MTree64_getPoints(self, selector, data)

    def size(self):
        return _dSalmon.MTree64_size(self)

    def dataDimension(self):
        return _dSalmon.MTree64_dataDimension(self)

    def clear(self):
        return _dSalmon.MTree64_clear(self)

    def serialize(self):
        return _dSalmon.MTree64_serialize(self)

    def unserialize(self, data):
        return _dSalmon.MTree64_unserialize(self, data)
    __swig_destroy__ = _dSalmon.delete_MTree64

# Register MTree64 in _dSalmon:
_dSalmon.MTree64_swigregister(MTree64)

class MTreeSelector32(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _dSalmon.MTreeSelector32_swiginit(self, _dSalmon.new_MTreeSelector32(*args))

    def allOk(self):
        return _dSalmon.MTreeSelector32_allOk(self)

    def getFoundMask(self, found):
        return _dSalmon.MTreeSelector32_getFoundMask(self, found)

    def size(self):
        return _dSalmon.MTreeSelector32_size(self)
    __swig_destroy__ = _dSalmon.delete_MTreeSelector32

# Register MTreeSelector32 in _dSalmon:
_dSalmon.MTreeSelector32_swigregister(MTreeSelector32)

class MTreeSelector64(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _dSalmon.MTreeSelector64_swiginit(self, _dSalmon.new_MTreeSelector64(*args))

    def allOk(self):
        return _dSalmon.MTreeSelector64_allOk(self)

    def getFoundMask(self, found):
        return _dSalmon.MTreeSelector64_getFoundMask(self, found)

    def size(self):
        return _dSalmon.MTreeSelector64_size(self)
    __swig_destroy__ = _dSalmon.delete_MTreeSelector64

# Register MTreeSelector64 in _dSalmon:
_dSalmon.MTreeSelector64_swigregister(MTreeSelector64)

class MTreeRangeQuery32(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, tree, data, radius):
        _dSalmon.MTreeRangeQuery32_swiginit(self, _dSalmon.new_MTreeRangeQuery32(tree, data, radius))

    def result(self, indices, distances):
        return _dSalmon.MTreeRangeQuery32_result(self, indices, distances)

    def resultLengths(self, lengths):
        return _dSalmon.MTreeRangeQuery32_resultLengths(self, lengths)

    def resultTotalLength(self):
        return _dSalmon.MTreeRangeQuery32_resultTotalLength(self)
    __swig_destroy__ = _dSalmon.delete_MTreeRangeQuery32

# Register MTreeRangeQuery32 in _dSalmon:
_dSalmon.MTreeRangeQuery32_swigregister(MTreeRangeQuery32)

class MTreeRangeQuery64(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, tree, data, radius):
        _dSalmon.MTreeRangeQuery64_swiginit(self, _dSalmon.new_MTreeRangeQuery64(tree, data, radius))

    def result(self, indices, distances):
        return _dSalmon.MTreeRangeQuery64_result(self, indices, distances)

    def resultLengths(self, lengths):
        return _dSalmon.MTreeRangeQuery64_resultLengths(self, lengths)

    def resultTotalLength(self):
        return _dSalmon.MTreeRangeQuery64_resultTotalLength(self)
    __swig_destroy__ = _dSalmon.delete_MTreeRangeQuery64

# Register MTreeRangeQuery64 in _dSalmon:
_dSalmon.MTreeRangeQuery64_swigregister(MTreeRangeQuery64)

class MTreeKnnQuery32(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, tree, data, k, sort, min_radius, max_radius, reverse, extend_for_ties):
        _dSalmon.MTreeKnnQuery32_swiginit(self, _dSalmon.new_MTreeKnnQuery32(tree, data, k, sort, min_radius, max_radius, reverse, extend_for_ties))

    def result(self, indices, distances):
        return _dSalmon.MTreeKnnQuery32_result(self, indices, distances)

    def resultLengths(self, lengths):
        return _dSalmon.MTreeKnnQuery32_resultLengths(self, lengths)

    def resultTotalLength(self):
        return _dSalmon.MTreeKnnQuery32_resultTotalLength(self)
    __swig_destroy__ = _dSalmon.delete_MTreeKnnQuery32

# Register MTreeKnnQuery32 in _dSalmon:
_dSalmon.MTreeKnnQuery32_swigregister(MTreeKnnQuery32)

class MTreeKnnQuery64(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, tree, data, k, sort, min_radius, max_radius, reverse, extend_for_ties):
        _dSalmon.MTreeKnnQuery64_swiginit(self, _dSalmon.new_MTreeKnnQuery64(tree, data, k, sort, min_radius, max_radius, reverse, extend_for_ties))

    def result(self, indices, distances):
        return _dSalmon.MTreeKnnQuery64_result(self, indices, distances)

    def resultLengths(self, lengths):
        return _dSalmon.MTreeKnnQuery64_resultLengths(self, lengths)

    def resultTotalLength(self):
        return _dSalmon.MTreeKnnQuery64_resultTotalLength(self)
    __swig_destroy__ = _dSalmon.delete_MTreeKnnQuery64

# Register MTreeKnnQuery64 in _dSalmon:
_dSalmon.MTreeKnnQuery64_swigregister(MTreeKnnQuery64)

class SDOstream32(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, observer_cnt, T, idle_observers, neighbour_cnt, freq_bins, max_freq, distance, seed):
        _dSalmon.SDOstream32_swiginit(self, _dSalmon.new_SDOstream32(observer_cnt, T, idle_observers, neighbour_cnt, freq_bins, max_freq, distance, seed))

    def fit(self, data, times):
        return _dSalmon.SDOstream32_fit(self, data, times)

    def fit_predict(self, data, scores, times):
        return _dSalmon.SDOstream32_fit_predict(self, data, scores, times)

    def fit_predict_with_sampling(self, data, scores, times, sampled):
        return _dSalmon.SDOstream32_fit_predict_with_sampling(self, data, scores, times, sampled)

    def observer_count(self):
        return _dSalmon.SDOstream32_observer_count(self)

    def frequency_bin_count(self):
        return _dSalmon.SDOstream32_frequency_bin_count(self)

    def get_observers(self, data, observations, av_observations, time):
        return _dSalmon.SDOstream32_get_observers(self, data, observations, av_observations, time)
    __swig_destroy__ = _dSalmon.delete_SDOstream32

# Register SDOstream32 in _dSalmon:
_dSalmon.SDOstream32_swigregister(SDOstream32)

class SDOstream64(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, observer_cnt, T, idle_observers, neighbour_cnt, freq_bins, max_freq, distance, seed):
        _dSalmon.SDOstream64_swiginit(self, _dSalmon.new_SDOstream64(observer_cnt, T, idle_observers, neighbour_cnt, freq_bins, max_freq, distance, seed))

    def fit(self, data, times):
        return _dSalmon.SDOstream64_fit(self, data, times)

    def fit_predict(self, data, scores, times):
        return _dSalmon.SDOstream64_fit_predict(self, data, scores, times)

    def fit_predict_with_sampling(self, data, scores, times, sampled):
        return _dSalmon.SDOstream64_fit_predict_with_sampling(self, data, scores, times, sampled)

    def observer_count(self):
        return _dSalmon.SDOstream64_observer_count(self)

    def frequency_bin_count(self):
        return _dSalmon.SDOstream64_frequency_bin_count(self)

    def get_observers(self, data, observations, av_observations, time):
        return _dSalmon.SDOstream64_get_observers(self, data, observations, av_observations, time)
    __swig_destroy__ = _dSalmon.delete_SDOstream64

# Register SDOstream64 in _dSalmon:
_dSalmon.SDOstream64_swigregister(SDOstream64)

class DBOR32(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, window, radius, distance, min_node_size, max_node_size, split_sampling):
        _dSalmon.DBOR32_swiginit(self, _dSalmon.new_DBOR32(window, radius, distance, min_node_size, max_node_size, split_sampling))

    def fit(self, data, times):
        return _dSalmon.DBOR32_fit(self, data, times)

    def fit_predict(self, data, scores, times):
        return _dSalmon.DBOR32_fit_predict(self, data, scores, times)

    def window_size(self):
        return _dSalmon.DBOR32_window_size(self)

    def get_window(self, data, times, neighbors):
        return _dSalmon.DBOR32_get_window(self, data, times, neighbors)
    __swig_destroy__ = _dSalmon.delete_DBOR32

# Register DBOR32 in _dSalmon:
_dSalmon.DBOR32_swigregister(DBOR32)

class DBOR64(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, window, radius, distance, min_node_size, max_node_size, split_sampling):
        _dSalmon.DBOR64_swiginit(self, _dSalmon.new_DBOR64(window, radius, distance, min_node_size, max_node_size, split_sampling))

    def fit(self, data, times):
        return _dSalmon.DBOR64_fit(self, data, times)

    def fit_predict(self, data, scores, times):
        return _dSalmon.DBOR64_fit_predict(self, data, scores, times)

    def window_size(self):
        return _dSalmon.DBOR64_window_size(self)

    def get_window(self, data, times, neighbors):
        return _dSalmon.DBOR64_get_window(self, data, times, neighbors)
    __swig_destroy__ = _dSalmon.delete_DBOR64

# Register DBOR64 in _dSalmon:
_dSalmon.DBOR64_swigregister(DBOR64)

class SWKNN32(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, window, neighbor_cnt, distance, min_node_size, max_node_size, split_sampling):
        _dSalmon.SWKNN32_swiginit(self, _dSalmon.new_SWKNN32(window, neighbor_cnt, distance, min_node_size, max_node_size, split_sampling))

    def fit(self, data, times):
        return _dSalmon.SWKNN32_fit(self, data, times)

    def fit_predict(self, data, scores, times):
        return _dSalmon.SWKNN32_fit_predict(self, data, scores, times)

    def fit_predict_with_neighbors(self, data, scores, times):
        return _dSalmon.SWKNN32_fit_predict_with_neighbors(self, data, scores, times)

    def window_size(self):
        return _dSalmon.SWKNN32_window_size(self)

    def get_window(self, data, times):
        return _dSalmon.SWKNN32_get_window(self, data, times)
    __swig_destroy__ = _dSalmon.delete_SWKNN32

# Register SWKNN32 in _dSalmon:
_dSalmon.SWKNN32_swigregister(SWKNN32)

class SWKNN64(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, window, neighbor_cnt, distance, min_node_size, max_node_size, split_sampling):
        _dSalmon.SWKNN64_swiginit(self, _dSalmon.new_SWKNN64(window, neighbor_cnt, distance, min_node_size, max_node_size, split_sampling))

    def fit(self, data, times):
        return _dSalmon.SWKNN64_fit(self, data, times)

    def fit_predict(self, data, scores, times):
        return _dSalmon.SWKNN64_fit_predict(self, data, scores, times)

    def fit_predict_with_neighbors(self, data, scores, times):
        return _dSalmon.SWKNN64_fit_predict_with_neighbors(self, data, scores, times)

    def window_size(self):
        return _dSalmon.SWKNN64_window_size(self)

    def get_window(self, data, times):
        return _dSalmon.SWKNN64_get_window(self, data, times)
    __swig_destroy__ = _dSalmon.delete_SWKNN64

# Register SWKNN64 in _dSalmon:
_dSalmon.SWKNN64_swigregister(SWKNN64)

class SWLOF32(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, window, neighbor_cnt, simplified, distance, min_node_size, max_node_size, split_sampling):
        _dSalmon.SWLOF32_swiginit(self, _dSalmon.new_SWLOF32(window, neighbor_cnt, simplified, distance, min_node_size, max_node_size, split_sampling))

    def fit(self, data, times):
        return _dSalmon.SWLOF32_fit(self, data, times)

    def fit_predict(self, data, scores, times):
        return _dSalmon.SWLOF32_fit_predict(self, data, scores, times)

    def window_size(self):
        return _dSalmon.SWLOF32_window_size(self)

    def get_window(self, data, times):
        return _dSalmon.SWLOF32_get_window(self, data, times)
    __swig_destroy__ = _dSalmon.delete_SWLOF32

# Register SWLOF32 in _dSalmon:
_dSalmon.SWLOF32_swigregister(SWLOF32)

class SWLOF64(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, window, neighbor_cnt, simplified, distance, min_node_size, max_node_size, split_sampling):
        _dSalmon.SWLOF64_swiginit(self, _dSalmon.new_SWLOF64(window, neighbor_cnt, simplified, distance, min_node_size, max_node_size, split_sampling))

    def fit(self, data, times):
        return _dSalmon.SWLOF64_fit(self, data, times)

    def fit_predict(self, data, scores, times):
        return _dSalmon.SWLOF64_fit_predict(self, data, scores, times)

    def window_size(self):
        return _dSalmon.SWLOF64_window_size(self)

    def get_window(self, data, times):
        return _dSalmon.SWLOF64_get_window(self, data, times)
    __swig_destroy__ = _dSalmon.delete_SWLOF64

# Register SWLOF64 in _dSalmon:
_dSalmon.SWLOF64_swigregister(SWLOF64)

class RRCT32(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, tree_cnt, window, seed, n_jobs):
        _dSalmon.RRCT32_swiginit(self, _dSalmon.new_RRCT32(tree_cnt, window, seed, n_jobs))

    def fit(self, data, times):
        return _dSalmon.RRCT32_fit(self, data, times)

    def fit_predict(self, data, scores, times):
        return _dSalmon.RRCT32_fit_predict(self, data, scores, times)

    def window_size(self):
        return _dSalmon.RRCT32_window_size(self)

    def get_window(self, data_out, times):
        return _dSalmon.RRCT32_get_window(self, data_out, times)
    __swig_destroy__ = _dSalmon.delete_RRCT32

# Register RRCT32 in _dSalmon:
_dSalmon.RRCT32_swigregister(RRCT32)

class RRCT64(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, tree_cnt, window, seed, n_jobs):
        _dSalmon.RRCT64_swiginit(self, _dSalmon.new_RRCT64(tree_cnt, window, seed, n_jobs))

    def fit(self, data, times):
        return _dSalmon.RRCT64_fit(self, data, times)

    def fit_predict(self, data, scores, times):
        return _dSalmon.RRCT64_fit_predict(self, data, scores, times)

    def window_size(self):
        return _dSalmon.RRCT64_window_size(self)

    def get_window(self, data_out, times):
        return _dSalmon.RRCT64_get_window(self, data_out, times)
    __swig_destroy__ = _dSalmon.delete_RRCT64

# Register RRCT64 in _dSalmon:
_dSalmon.RRCT64_swigregister(RRCT64)

class SWHBOS32(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, window, bins):
        _dSalmon.SWHBOS32_swiginit(self, _dSalmon.new_SWHBOS32(window, bins))

    def fit(self, data, times):
        return _dSalmon.SWHBOS32_fit(self, data, times)

    def fit_predict(self, data, scores, times):
        return _dSalmon.SWHBOS32_fit_predict(self, data, scores, times)

    def window_size(self):
        return _dSalmon.SWHBOS32_window_size(self)

    def get_window(self, data, times):
        return _dSalmon.SWHBOS32_get_window(self, data, times)
    __swig_destroy__ = _dSalmon.delete_SWHBOS32

# Register SWHBOS32 in _dSalmon:
_dSalmon.SWHBOS32_swigregister(SWHBOS32)

class SWHBOS64(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, window, bins):
        _dSalmon.SWHBOS64_swiginit(self, _dSalmon.new_SWHBOS64(window, bins))

    def fit(self, data, times):
        return _dSalmon.SWHBOS64_fit(self, data, times)

    def fit_predict(self, data, scores, times):
        return _dSalmon.SWHBOS64_fit_predict(self, data, scores, times)

    def window_size(self):
        return _dSalmon.SWHBOS64_window_size(self)

    def get_window(self, data, times):
        return _dSalmon.SWHBOS64_get_window(self, data, times)
    __swig_destroy__ = _dSalmon.delete_SWHBOS64

# Register SWHBOS64 in _dSalmon:
_dSalmon.SWHBOS64_swigregister(SWHBOS64)



