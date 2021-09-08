// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

#ifndef DSALMON_DISTANCE_WRAPPERS_H
#define DSALMON_DISTANCE_WRAPPERS_H

#include <functional>

#include "Vector.h"
#include "array_types.h"

// TODO:  for p < 1, lp is not a metric!!


#ifdef SWIG
%ignore *::getFunction();
#endif

template<typename FloatType>
class Distance_wrapper {
  protected:
    typedef std::function<FloatType(const Vector<FloatType>&, const Vector<FloatType>&)> DistanceFunction;
    // protect ctor and dtor, so that SWIG doesn't create
    // new and delete wrappers for this class
    Distance_wrapper() = default;
    virtual ~Distance_wrapper() {};
  public:
    virtual DistanceFunction getFunction() = 0;
};
DEFINE_FLOATINSTANTIATIONS(Distance)

template<typename FloatType>
class EuclideanDist_wrapper : public Distance_wrapper<FloatType> {
  public:
    typename Distance_wrapper<FloatType>::DistanceFunction getFunction() override  {
        return Vector<FloatType>::euclidean;
    };
};
DEFINE_FLOATINSTANTIATIONS(EuclideanDist)

template<typename FloatType>
class ManhattanDist_wrapper : public Distance_wrapper<FloatType> {
  public:
    typename Distance_wrapper<FloatType>::DistanceFunction getFunction() override {
        return Vector<FloatType>::manhattan;
    }
};
DEFINE_FLOATINSTANTIATIONS(ManhattanDist)

template<typename FloatType>
class ChebyshevDist_wrapper : public Distance_wrapper<FloatType> {
  public:
    typename Distance_wrapper<FloatType>::DistanceFunction getFunction() override {
        return Vector<FloatType>::chebyshev;
    }
};
DEFINE_FLOATINSTANTIATIONS(ChebyshevDist)

template<typename FloatType>
class MinkowskiDist_wrapper : public Distance_wrapper<FloatType> {
    int p;
  public:
    MinkowskiDist_wrapper(int p) : p(p) {}
    typename Distance_wrapper<FloatType>::DistanceFunction getFunction() override {
        return std::bind(
            Vector<FloatType>::lp, 
            std::placeholders::_1,
            std::placeholders::_2,
            p
        );
    }
};
DEFINE_FLOATINSTANTIATIONS(MinkowskiDist)

#endif