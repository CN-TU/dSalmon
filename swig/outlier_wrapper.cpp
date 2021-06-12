// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

#include <algorithm>
#include <vector>
#include <assert.h>
#include <random>
#include <atomic>
#include <thread>

#include "outlier_wrapper.h"

template<typename FloatType>
SDOstream_wrapper<FloatType>::SDOstream_wrapper(int observer_cnt, FloatType T, FloatType idle_observers, int neighbour_cnt, int freq_bins, FloatType max_freq, Distance_wrapper<FloatType>* distance, int seed) :
	dimension(-1),
	freq_bins(1),
	// freq_bins(freq_bins), // use freq_bins and max_freq parameters when implementing periodic SDOstream
	sdo(observer_cnt, T, idle_observers, neighbour_cnt, distance->getFunction(), seed)
	// sdo(observer_cnt, T, idle_observers, neighbour_cnt, freq_bins, max_freq, distance->getFunction(), seed)
{
}

template<typename FloatType>
void SDOstream_wrapper<FloatType>::fit(const NumpyArray2<FloatType> data, const NumpyArray1<FloatType> times) {
	for (int i = 0; i < data.dim1; i++) {
		sdo.fit(Vector<FloatType>{&data.data[i * data.dim2], data.dim2}, times.data[i]);
	}
}

template<typename FloatType>
void SDOstream_wrapper<FloatType>::fit_predict(const NumpyArray2<FloatType> data, NumpyArray1<FloatType> scores, const NumpyArray1<FloatType> times) {
	for (int i = 0; i < data.dim1; i++) {
		scores.data[i] = sdo.fitPredict(Vector<FloatType>{&data.data[i * data.dim2], data.dim2}, times.data[i]);
	}
}

template<typename FloatType>
void SDOstream_wrapper<FloatType>::fit_predict_with_sampling(const NumpyArray2<FloatType> data, NumpyArray1<FloatType> scores, const NumpyArray1<FloatType> times, NumpyArray1<int> sampled) {
	for (int i = 0; i < data.dim1; i++) {
		scores.data[i] = sdo.fitPredict(Vector<FloatType>{&data.data[i * data.dim2], data.dim2}, times.data[i]);
		sampled.data[i] = sdo.lastWasSampled();
	}
}

template<typename FloatType>
int SDOstream_wrapper<FloatType>::observer_count() {
	return sdo.observerCount();
}

template<typename FloatType>
int SDOstream_wrapper<FloatType>::frequency_bin_count() {
	return freq_bins;
}

template<typename FloatType>
void SDOstream_wrapper<FloatType>::get_observers(NumpyArray2<FloatType> data, NumpyArray2<std::complex<FloatType>> observations, NumpyArray1<FloatType> av_observations, FloatType time) {
	// TODO: check dimensions
	int i = 0;
	for (auto observer : sdo) {
		Vector<FloatType> vec_data = observer.getData();
		std::copy(vec_data.begin(), vec_data.end(), &data.data[i * data.dim2]);
		observations.data[i] = observer.getObservations(time);
		//TODO: use complex ft when implementing periodic SDOstream
		//std::vector<std::complex<FloatType>> observations_ft = observer.getObservations(time);
		//std::copy(observations_ft.begin(), observations_ft.end(), &observations.data[i * observations.dim2]);
		av_observations.data[i] = observer.getAvObservations(time);
		i++;
	}
}


template<typename FloatType>
DBOR_wrapper<FloatType>::DBOR_wrapper(FloatType window, FloatType radius, Distance_wrapper<FloatType>* distance, int min_node_size, int max_node_size, int split_sampling) :
	dbor(window, radius, distance->getFunction(), min_node_size, max_node_size, split_sampling)
{ }

template<typename FloatType>
void DBOR_wrapper<FloatType>::fit(const NumpyArray2<FloatType> data, const NumpyArray1<FloatType> times) {
	assert (data.dim1 == times.dim1);
	for (int i = 0; i < data.dim1; i++) {
		dbor.fit(Vector<FloatType>{&data.data[i * data.dim2], data.dim2}, times.data[i]);
	}
}

template<typename FloatType>
void DBOR_wrapper<FloatType>::fit_predict(const NumpyArray2<FloatType> data, NumpyArray1<FloatType> scores, const NumpyArray1<FloatType> times) {
	assert (data.dim1 == times.dim1);
	assert (data.dim1 == scores.dim1);
	for (int i = 0; i < data.dim1; i++) {
		scores.data[i] = dbor.fitPredict(Vector<FloatType>{&data.data[i * data.dim2], data.dim2}, times.data[i]);
	}
}

template<typename FloatType>
int DBOR_wrapper<FloatType>::window_size() {
	return dbor.windowSize();
}

template<typename FloatType>
void DBOR_wrapper<FloatType>::get_window(NumpyArray2<FloatType> data, NumpyArray1<FloatType> times, NumpyArray1<int> neighbors) {
	int i = 0;
	for (auto sample : dbor) {
		Vector<FloatType> vec_data = sample.getSample();
		std::copy(vec_data.begin(), vec_data.end(), &data.data[i * data.dim2]);
		times.data[i] = sample.getExpireTime();
		neighbors.data[i] = sample.getNeighbors();
		i++;
	}
}


template<typename FloatType>
SWKNN_wrapper<FloatType>::SWKNN_wrapper(FloatType window, int neighbor_cnt, Distance_wrapper<FloatType>* distance, int min_node_size, int max_node_size, int split_sampling) :
	swknn(window, neighbor_cnt, distance->getFunction(), min_node_size, max_node_size, split_sampling)
{ }

template<typename FloatType>
void SWKNN_wrapper<FloatType>::fit(const NumpyArray2<FloatType> data, const NumpyArray1<FloatType> times) {
	assert (data.dim1 == times.dim1);
	for (int i = 0; i < data.dim1; i++) {
		swknn.fit(Vector<FloatType>{&data.data[i * data.dim2], data.dim2}, times.data[i]);
	}
}

template<typename FloatType>
void SWKNN_wrapper<FloatType>::fit_predict(const NumpyArray2<FloatType> data, NumpyArray1<FloatType> scores, const NumpyArray1<FloatType> times) {
	assert (data.dim1 == times.dim1);
	assert (data.dim1 == scores.dim1);
	for (int i = 0; i < data.dim1; i++) {
		scores.data[i] = swknn.fitPredict(Vector<FloatType>{&data.data[i * data.dim2], data.dim2}, times.data[i]);
	}
}

template<typename FloatType>
void SWKNN_wrapper<FloatType>::fit_predict_with_neighbors(const NumpyArray2<FloatType> data, NumpyArray2<FloatType> scores, const NumpyArray1<FloatType> times) {
	assert (data.dim1 == times.dim1);
	assert (data.dim1 == scores.dim1);
	for (int i = 0; i < data.dim1; i++) {
		std::vector<FloatType> scores_vec =
			swknn.fitPredictWithNeighbors(Vector<FloatType>{&data.data[i * data.dim2], data.dim2}, times.data[i]);
		std::copy(scores_vec.begin(), scores_vec.end(), &scores.data[i * scores.dim2]);
	}
}

template<typename FloatType>
int SWKNN_wrapper<FloatType>::window_size() {
	return swknn.windowSize();
}

template<typename FloatType>
void SWKNN_wrapper<FloatType>::get_window(NumpyArray2<FloatType> data, NumpyArray1<FloatType> times) {
	int i = 0;
	for (auto sample : swknn) {
		Vector<FloatType> vec_data = sample.getSample();
		std::copy(vec_data.begin(), vec_data.end(), &data.data[i * data.dim2]);
		times.data[i] = sample.getExpireTime();
		i++;
	}
}


template<typename FloatType>
SWLOF_wrapper<FloatType>::SWLOF_wrapper(FloatType window, int neighbor_cnt, bool simplified, Distance_wrapper<FloatType>* distance, int min_node_size, int max_node_size, int split_sampling) :
	swlof(window, neighbor_cnt, simplified, distance->getFunction(), min_node_size, max_node_size, split_sampling)
{ }

template<typename FloatType>
void SWLOF_wrapper<FloatType>::fit(const NumpyArray2<FloatType> data, const NumpyArray1<FloatType> times) {
	assert (data.dim1 == times.dim1);
	for (int i = 0; i < data.dim1; i++) {
		swlof.fit(Vector<FloatType>{&data.data[i * data.dim2], data.dim2}, times.data[i]);
	}
}

template<typename FloatType>
void SWLOF_wrapper<FloatType>::fit_predict(const NumpyArray2<FloatType> data, NumpyArray2<FloatType> scores, const NumpyArray1<FloatType> times) {
	assert (data.dim1 == times.dim1);
	assert (data.dim1 == scores.dim1);
	for (int i = 0; i < data.dim1; i++) {
		std::vector<FloatType> scores_vec = swlof.fitPredict(Vector<FloatType>{&data.data[i * data.dim2], data.dim2}, times.data[i]);
		std::copy(scores_vec.begin(), scores_vec.end(), &scores.data[i * scores.dim2]);
	}
}

template<typename FloatType>
int SWLOF_wrapper<FloatType>::window_size() {
	return swlof.windowSize();
}

template<typename FloatType>
void SWLOF_wrapper<FloatType>::get_window(NumpyArray2<FloatType> data, NumpyArray1<FloatType> times) {
	int i = 0;
	for (auto sample : swlof) {
		Vector<FloatType> vec_data = sample.getSample();
		std::copy(vec_data.begin(), vec_data.end(), &data.data[i * data.dim2]);
		times.data[i] = sample.getExpireTime();
		i++;
	}
}


template<typename FloatType>
RRCT_wrapper<FloatType>::RRCT_wrapper(unsigned tree_cnt, FloatType window, int seed, unsigned n_jobs) :
	window(window), n_jobs(std::min(tree_cnt, n_jobs))
{
	rrct.reserve(tree_cnt);
	std::mt19937 rng(seed);
	for (unsigned i = 0; i < tree_cnt; i++)
		rrct.emplace_back(rng());
}

template<typename FloatType>
void RRCT_wrapper<FloatType>::pruneExpired(RRCT<FloatType,FloatType>& tree, FloatType now) {
	while (!tree.empty() && tree.front().second <= now)
		tree.pop_front();
}

template<typename FloatType>
void RRCT_wrapper<FloatType>::fit(const NumpyArray2<FloatType> data, const NumpyArray1<FloatType> times) {
	fit_predict(data, {nullptr,0}, times);
}

template<typename FloatType>
static void fit_predict_ensemble(const NumpyArray2<FloatType>& data, unsigned ensemble_size, int n_jobs, std::function<void(const std::vector<std::shared_ptr<Vector<FloatType>>>&,int)> worker) {
	std::vector<std::shared_ptr<Vector<FloatType>>> vecs;
	vecs.reserve(data.dim1);
	for (int i = 0; i < data.dim1; i++)
		vecs.emplace_back(new Vector<FloatType>(&data.data[i * data.dim2], data.dim2));
	std::atomic<unsigned> global_i(0);
	auto thread_worker = [&]() {
		for (unsigned tree_index = global_i++; tree_index < ensemble_size; tree_index = global_i++) {
			worker(vecs, tree_index);
		}
	};
	if (n_jobs < 2) {
		thread_worker();
	}
	else {
		std::thread threads[n_jobs];
		for (int i = 0; i < n_jobs; i++)
			threads[i] = std::thread{thread_worker};
		for (int i = 0; i < n_jobs; i++)
			threads[i].join();
	}
}

template<typename FloatType>
void RRCT_wrapper<FloatType>::fit_predict(const NumpyArray2<FloatType> data, NumpyArray1<FloatType> scores, const NumpyArray1<FloatType> times) {
	assert (data.dim1 == times.dim1);
	assert ((scores.data == nullptr && scores.dim1 == 0) || data.dim1 == scores.dim1);
	bool fit_only = scores.data == nullptr;
	std::vector<std::mutex> locks(fit_only ? 0 : data.dim1);
	if (!fit_only)
		std::fill(scores.data, scores.data + data.dim1, 0);
	auto worker = [&](const std::vector<std::shared_ptr<Vector<FloatType>>>& vecs, int tree_index) {
		auto& tree = rrct[tree_index];
		for (std::size_t i = 0; i < vecs.size(); i++) {
			pruneExpired(tree, times.data[i]);
			tree.push_back(std::make_pair(vecs[i], times.data[i] + window));
			if (!fit_only) {
				FloatType codisp = tree.codisp(std::prev(tree.end()));
				locks[i].lock();
				scores.data[i] += codisp;
				locks[i].unlock();
			}
		}
	};
	fit_predict_ensemble<FloatType>(data, rrct.size(), n_jobs, worker);
	if (!fit_only) {
		// TODO: check if this is really average
		for (int i = 0; i < data.dim1; i++)
			scores.data[i] /= rrct.size();
	}
}

template<typename FloatType>
int RRCT_wrapper<FloatType>::window_size() {
	return rrct[0].size();
}

template<typename FloatType>
void RRCT_wrapper<FloatType>::get_window(NumpyArray2<FloatType> data, NumpyArray1<FloatType> times) {
	int i = 0;
	for (auto& sample : rrct[0]) {
		const Vector<FloatType>& vec_data = *sample.first;
		std::copy(vec_data.begin(), vec_data.end(), &data.data[i * data.dim2]);
		times.data[i] = sample.second;
		i++;
	}
}


template<typename FloatType>
RSHash_wrapper<FloatType>::RSHash_wrapper(unsigned ensemble_size, FloatType window, int cms_w_param, int cms_d_param, FloatType s_param, int seed, unsigned n_jobs) :
	n_jobs(n_jobs)
{
	ensemble.reserve(ensemble_size);
	std::mt19937 rng(seed);
	// Vector<FloatType> min_estimates_vec{min_estimates.data, min_estimates.dim1};
	// Vector<FloatType> max_estimates_vec{max_estimates.data, max_estimates.dim1};
	for (unsigned i = 0; i < ensemble_size; i++)
		ensemble.emplace_back(window, s_param, cms_w_param, cms_d_param, rng());
}


template<typename FloatType>
void RSHash_wrapper<FloatType>::fit_predict(const NumpyArray2<FloatType> data, NumpyArray1<FloatType> scores, const NumpyArray1<FloatType> times) {
	assert (data.dim1 == times.dim1);
	assert ((scores.data == nullptr && scores.dim1 == 0) || data.dim1 == scores.dim1);
	bool fit_only = scores.data == nullptr;
	std::vector<std::mutex> locks(fit_only ? 0 : data.dim1);
	if (!fit_only)
		std::fill(scores.data, scores.data + data.dim1, 0);
	auto worker = [&](const std::vector<std::shared_ptr<Vector<FloatType>>>& vecs, int ensemble_index) {
		auto& detector = ensemble[ensemble_index];
		for (std::size_t i = 0; i < vecs.size(); i++) {
			FloatType score = detector.fitPredict(vecs[i], times.data[i]);
			if (!fit_only) {
				locks[i].lock();
				scores.data[i] += score;
				locks[i].unlock();
			}
		}
	};
	fit_predict_ensemble<FloatType>(data, ensemble.size(), n_jobs, worker);
	if (!fit_only) {
		for (int i = 0; i < data.dim1; i++)
			scores.data[i] /= ensemble.size();
	}
}

template<typename FloatType>
void RSHash_wrapper<FloatType>::fit(const NumpyArray2<FloatType> data, const NumpyArray1<FloatType> times) {
	fit_predict(data, {nullptr,0}, times);
}

template<typename FloatType>
int RSHash_wrapper<FloatType>::window_size() {
	return ensemble[0].windowSize();
}

template<typename FloatType>
void RSHash_wrapper<FloatType>::get_window(NumpyArray2<FloatType> data, NumpyArray1<FloatType> times) {
	int i = 0;
	for (auto sample : ensemble[0]) {
		Vector<FloatType> vec_data = sample.getSample();
		std::copy(vec_data.begin(), vec_data.end(), &data.data[i * data.dim2]);
		times.data[i] = sample.getExpireTime();
		i++;
	}
}



template<typename FloatType>
SWHBOS_wrapper<FloatType>::SWHBOS_wrapper(FloatType window, unsigned n_bins) :
	estimator(window, n_bins)
{
}

template<typename FloatType>
void SWHBOS_wrapper<FloatType>::fit(const NumpyArray2<FloatType> data, const NumpyArray1<FloatType> times) {
	assert (data.dim1 == times.dim1);
	for (int i = 0; i < data.dim1; i++) {
		estimator.append(Vector<FloatType>(&data.data[i * data.dim2], data.dim2), times.data[i]);
	}
}

template<typename FloatType>
void SWHBOS_wrapper<FloatType>::fit_predict(const NumpyArray2<FloatType> data, NumpyArray1<FloatType> scores, const NumpyArray1<FloatType> times) {
	assert (data.dim1 == times.dim1);
	assert (data.dim1 == scores.dim1);
	for (int i = 0; i < data.dim1; i++) {
		auto it = estimator.append(Vector<FloatType>(&data.data[i * data.dim2], data.dim2), times.data[i]);
		scores.data[i] = estimator.outlierness(it);
	}
}

template<typename FloatType>
int SWHBOS_wrapper<FloatType>::window_size() {
	return estimator.windowSize();
}

template<typename FloatType>
void SWHBOS_wrapper<FloatType>::get_window(NumpyArray2<FloatType> data, NumpyArray1<FloatType> times) {
	int i = 0;
	for (auto sample : estimator) {
		const Vector<FloatType>& vec_data = sample.getSample();
		std::copy(vec_data.begin(), vec_data.end(), &data.data[i * data.dim2]);
		times.data[i] = sample.getExpireTime();
		i++;
	}
}

template class SDOstream_wrapper<double>;
template class SDOstream_wrapper<float>;
template class DBOR_wrapper<double>;
template class DBOR_wrapper<float>;
template class SWKNN_wrapper<double>;
template class SWKNN_wrapper<float>;
template class SWLOF_wrapper<double>;
template class SWLOF_wrapper<float>;
template class RRCT_wrapper<double>;
template class RRCT_wrapper<float>;
template class RSHash_wrapper<double>;
template class RSHash_wrapper<float>;
template class SWHBOS_wrapper<double>;
template class SWHBOS_wrapper<float>;
