// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

#ifndef DSALMON_PREPROC_WRAPPER_H

#include "array_types.h"
#include "preproc.h"

template<typename FloatType>
class SWZScoreScaler_wrapper {
	SWZScoreScaler<FloatType> scaler;

public:
	SWZScoreScaler_wrapper(FloatType window);
	void transform(const NumpyArray2<FloatType> data, NumpyArray2<FloatType> normalized, const NumpyArray1<FloatType> times);
};
DEFINE_FLOATINSTANTIATIONS(SWZScoreScaler)


template<typename FloatType>
class SWQuantileScaler_wrapper {
	SWQuantileScaler<FloatType> scaler;

public:
	SWQuantileScaler_wrapper(FloatType window, FloatType quantile);
	void transform(const NumpyArray2<FloatType> data, NumpyArray2<FloatType> normalized, const NumpyArray1<FloatType> times);
};
DEFINE_FLOATINSTANTIATIONS(SWQuantileScaler)

#endif
