// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

#ifndef DSALMON_OUTLIER_WRAPPER_H
#define DSALMON_OUTLIER_WRAPPER_H

#include <complex>

#include "SDOstream.h"
#include "slidingWindow.h"
#include "rrct.h"
#include "rshash.h"
#include "array_types.h"
#include "distance_wrappers.h"

template<typename FloatType>
class SDOstream_wrapper {
	int dimension;
	std::size_t freq_bins;
	SDOstream<FloatType> sdo;
	
  public:
	SDOstream_wrapper(int observer_cnt, FloatType T, FloatType idle_observers, int neighbour_cnt, int freq_bins, FloatType max_freq, Distance_wrapper<FloatType>* distance, int seed);
	void fit(const NumpyArray2<FloatType> data, const NumpyArray1<FloatType> times);
	void fit_predict(const NumpyArray2<FloatType> data, NumpyArray1<FloatType> scores, const NumpyArray1<FloatType> times);
	void fit_predict_with_sampling(const NumpyArray2<FloatType> data, NumpyArray1<FloatType> scores, const NumpyArray1<FloatType> times, NumpyArray1<int> sampled);
	int observer_count();
	int frequency_bin_count();
	void get_observers(NumpyArray2<FloatType> data, NumpyArray2<std::complex<FloatType>> observations, NumpyArray1<FloatType> av_observations, FloatType time);
};
DEFINE_FLOATINSTANTIATIONS(SDOstream)


template<typename FloatType>
class DBOR_wrapper {
	int dimension;
	DBOR<FloatType> dbor;
	
  public:
	DBOR_wrapper(FloatType window, FloatType radius, Distance_wrapper<FloatType>* distance, int min_node_size, int max_node_size, int split_sampling);
	void fit(const NumpyArray2<FloatType> data, const NumpyArray1<FloatType> times);
	void fit_predict(const NumpyArray2<FloatType> data, NumpyArray1<FloatType> scores, const NumpyArray1<FloatType> times);
	int window_size();
	void get_window(NumpyArray2<FloatType> data, NumpyArray1<FloatType> times, NumpyArray1<int> neighbors);
};
DEFINE_FLOATINSTANTIATIONS(DBOR)


template<typename FloatType>
class SWKNN_wrapper {
	int dimension;
	SWKNN<FloatType> swknn;
	
  public:
	SWKNN_wrapper(FloatType window, int neighbor_cnt, Distance_wrapper<FloatType>* distance, int min_node_size, int max_node_size, int split_sampling);
	void fit(const NumpyArray2<FloatType> data, const NumpyArray1<FloatType> times);
	void fit_predict(const NumpyArray2<FloatType> data, NumpyArray1<FloatType> scores, const NumpyArray1<FloatType> times);
	void fit_predict_with_neighbors(const NumpyArray2<FloatType> data, NumpyArray2<FloatType> scores, const NumpyArray1<FloatType> times);
	int window_size();
	void get_window(NumpyArray2<FloatType> data, NumpyArray1<FloatType> times);
};
DEFINE_FLOATINSTANTIATIONS(SWKNN)

template<typename FloatType>
class SWLOF_wrapper {
	int dimension;
	SWLOF<FloatType> swlof;
	
  public:
	SWLOF_wrapper(FloatType window, int neighbor_cnt, bool simplified, Distance_wrapper<FloatType>* distance, int min_node_size, int max_node_size, int split_sampling);
	void fit(const NumpyArray2<FloatType> data, const NumpyArray1<FloatType> times);
	void fit_predict(const NumpyArray2<FloatType> data, NumpyArray2<FloatType> scores, const NumpyArray1<FloatType> times);
	int window_size();
	void get_window(NumpyArray2<FloatType> data, NumpyArray1<FloatType> times);
};
DEFINE_FLOATINSTANTIATIONS(SWLOF)


template<typename FloatType>
class RRCT_wrapper {
	std::vector<RRCT<FloatType,FloatType>> rrct;
	FloatType window;
	int n_jobs;
	void pruneExpired(RRCT<FloatType,FloatType>& tree, FloatType now);

  public:
	RRCT_wrapper(unsigned tree_cnt, FloatType window, int seed, unsigned n_jobs);
	void fit(const NumpyArray2<FloatType> data, const NumpyArray1<FloatType> times);
	void fit_predict(const NumpyArray2<FloatType> data, NumpyArray1<FloatType> scores, const NumpyArray1<FloatType> times);
	int window_size();
	void get_window(NumpyArray2<FloatType> data_out, NumpyArray1<FloatType> times);
};
DEFINE_FLOATINSTANTIATIONS(RRCT)

template<typename FloatType>
class RSHash_wrapper {
	std::vector<RSHash<FloatType>> ensemble;
	int n_jobs;

  public:
	RSHash_wrapper(unsigned ensemble_size, FloatType window, int cms_w_param, int cms_d_param, unsigned s_param, int seed, unsigned n_jobs);
	void fit(const NumpyArray2<FloatType> data, const NumpyArray1<FloatType> times);
	void fit_predict(const NumpyArray2<FloatType> data, NumpyArray1<FloatType> scores, const NumpyArray1<FloatType> times);
	int window_size();
	void get_window(NumpyArray2<FloatType> data_out, NumpyArray1<FloatType> times);
};
DEFINE_FLOATINSTANTIATIONS(RSHash)

template<typename FloatType>
class SWHBOS_wrapper {
	SWHBOS<FloatType> estimator;

  public:
	SWHBOS_wrapper(FloatType window, unsigned bins);
	void fit(const NumpyArray2<FloatType> data, const NumpyArray1<FloatType> times);
	void fit_predict(const NumpyArray2<FloatType> data, NumpyArray1<FloatType> scores, const NumpyArray1<FloatType> times);
	int window_size();
	void get_window(NumpyArray2<FloatType> data, NumpyArray1<FloatType> times);
};
DEFINE_FLOATINSTANTIATIONS(SWHBOS)

#endif
