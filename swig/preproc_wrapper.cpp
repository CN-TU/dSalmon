// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

#include "preproc_wrapper.h"

template<typename FloatType>
SWZScoreScaler_wrapper<FloatType>::SWZScoreScaler_wrapper(FloatType window) :
	scaler(window)
{ }

template<typename FloatType>
void SWZScoreScaler_wrapper<FloatType>::transform(const NumpyArray2<FloatType> data, NumpyArray2<FloatType> data_normalized, const NumpyArray1<FloatType> times) {
	assert (data.dim1 == times.dim1);
	assert (data_normalized.dim1 == data.dim1);
	assert (data_normalized.dim2 == data.dim2);
	for (int i = 0; i < data.dim1; i++) {
		Vector<FloatType> normalized = scaler.transform(Vector<FloatType>{&data.data[i * data.dim2], data.dim2}, times.data[i]);
		std::copy(&*normalized.begin(), &*normalized.end(), &data_normalized.data[i * data_normalized.dim2]);
	}
}


template<typename FloatType>
SWQuantileScaler_wrapper<FloatType>::SWQuantileScaler_wrapper(FloatType window, FloatType quantile) :
	scaler(window, quantile)
{ }

template<typename FloatType>
void SWQuantileScaler_wrapper<FloatType>::transform(const NumpyArray2<FloatType> data, NumpyArray2<FloatType> data_normalized, const NumpyArray1<FloatType> times) {
	assert (data.dim1 == times.dim1);
	assert (data_normalized.dim1 == data.dim1);
	assert (data_normalized.dim2 == data.dim2);
	for (int i = 0; i < data.dim1; i++) {
		Vector<FloatType> normalized = scaler.transform(Vector<FloatType>{&data.data[i * data.dim2], data.dim2}, times.data[i]);
		std::copy(&*normalized.begin(), &*normalized.end(), &data_normalized.data[i * data_normalized.dim2]);
	}
}

template class SWZScoreScaler_wrapper<double>;
template class SWZScoreScaler_wrapper<float>;
template class SWQuantileScaler_wrapper<double>;
template class SWQuantileScaler_wrapper<float>;
