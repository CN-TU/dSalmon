// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

#include <algorithm>
#include <vector>
#include <assert.h>
#include <random>
#include <atomic>
#include <thread>

#include <boost/functional/hash.hpp>

#include "outlier_wrapper.h"


template<typename FloatType>
static void fit_predict_ensemble(unsigned ensemble_size, int n_jobs, std::function<void(int)> worker) {
    std::atomic<unsigned> global_i(0);
    auto thread_worker = [&]() {
        for (unsigned tree_index = global_i++; tree_index < ensemble_size; tree_index = global_i++) {
            worker(tree_index);
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
SDOstream_wrapper<FloatType>::SDOstream_wrapper(int observer_cnt, FloatType T, FloatType idle_observers, int neighbour_cnt, Distance_wrapper<FloatType>* distance, int seed) :
    dimension(-1),
    sdo(observer_cnt, T, idle_observers, neighbour_cnt, distance->getFunction(), seed)
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
void SDOstream_wrapper<FloatType>::get_observers(NumpyArray2<FloatType> data, NumpyArray2<FloatType> observations, NumpyArray1<FloatType> av_observations, FloatType time) {
    // TODO: check dimensions
    int i = 0;
    for (auto observer : sdo) {
        Vector<FloatType> vec_data = observer.getData();
        std::copy(vec_data.begin(), vec_data.end(), &data.data[i * data.dim2]);
        observations.data[i] = observer.getObservations(time);
        av_observations.data[i] = observer.getAvObservations(time);
        i++;
    }
}


template<typename FloatType>
DBOR_wrapper<FloatType>::DBOR_wrapper(FloatType window, FloatType radius, Distance_wrapper<FloatType>* distance, int min_node_size, int max_node_size, int split_sampling) :
    dbor(window, radius, distance->getFunction(), min_node_size, max_node_size, split_sampling)
{ }

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
void RRCT_wrapper<FloatType>::fit_predict(const NumpyArray2<FloatType> data, NumpyArray1<FloatType> scores, const NumpyArray1<FloatType> times) {
    assert (data.dim1 == times.dim1);
    assert ((scores.data == nullptr && scores.dim1 == 0) || data.dim1 == scores.dim1);
    bool fit_only = scores.data == nullptr;
    std::vector<std::mutex> locks(fit_only ? 0 : data.dim1);
    std::vector<std::shared_ptr<Vector<FloatType>>> vecs;
    vecs.reserve(data.dim1);
    for (int i = 0; i < data.dim1; i++)
        vecs.emplace_back(new Vector<FloatType>(&data.data[i * data.dim2], data.dim2));
    if (!fit_only)
        std::fill(scores.data, scores.data + data.dim1, 0);
    auto worker = [&](int tree_index) {
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
    fit_predict_ensemble<FloatType>(rrct.size(), n_jobs, worker);
    if (!fit_only) {
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
    for (unsigned i = 0; i < ensemble_size; i++)
        ensemble.emplace_back(window, s_param, cms_w_param, cms_d_param, rng());
}


template<typename FloatType>
void RSHash_wrapper<FloatType>::fit_predict(const NumpyArray2<FloatType> data, NumpyArray1<FloatType> scores, const NumpyArray1<FloatType> times) {
    bool fit_only = scores.data == nullptr;
    assert (data.dim1 == times.dim1);
    assert ((fit_only && scores.dim1 == 0) || data.dim1 == scores.dim1);
    
    std::vector<std::mutex> locks(fit_only ? 0 : data.dim1);
    std::vector<std::shared_ptr<Vector<FloatType>>> vecs;
    vecs.reserve(data.dim1);
    for (int i = 0; i < data.dim1; i++)
        vecs.emplace_back(new Vector<FloatType>(&data.data[i * data.dim2], data.dim2));
    if (!fit_only)
        std::fill(scores.data, scores.data + data.dim1, 0);
    auto worker = [&](int ensemble_index) {
        auto& detector = ensemble[ensemble_index];
        for (std::size_t i = 0; i < vecs.size(); i++) {
            FloatType score = detector.fitPredict(vecs[i], times.data[i]);
            if (!fit_only) {
                locks[i].lock();
                scores.data[i] += score / ensemble.size();
                locks[i].unlock();
            }
        }
    };
    fit_predict_ensemble<FloatType>(ensemble.size(), n_jobs, worker);
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
SWHBOS_wrapper<FloatType>::SWHBOS_wrapper(FloatType window, unsigned n_bins, unsigned n_jobs) :
    estimator(window, n_bins),
    n_jobs(n_jobs)
{ }

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
    if (n_jobs > 1 && data.dim1 > 1)
        estimator.initThreadPool(std::min<unsigned>(data.dim2, n_jobs));
    for (int i = 0; i < data.dim1; i++) {
        auto it = estimator.append(Vector<FloatType>(&data.data[i * data.dim2], data.dim2), times.data[i]);
        scores.data[i] = estimator.outlierness(it);
    }
    if (n_jobs > 1)
        estimator.releaseThreadPool();
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

template<typename FloatType>
HSTrees_wrapper<FloatType>::HSTrees_wrapper(FloatType window, unsigned tree_count, unsigned max_depth, unsigned size_limit, int seed, unsigned n_jobs) :
    n_jobs(n_jobs)
{
    ensemble.reserve(tree_count);
    std::mt19937 rng(seed);
    for (unsigned i = 0; i < tree_count; i++)
        ensemble.emplace_back(window, max_depth, size_limit, rng());
}

template<typename FloatType>
void HSTrees_wrapper<FloatType>::fit_predict(const NumpyArray2<FloatType> data, NumpyArray1<FloatType> scores, const NumpyArray1<FloatType> times) {
    bool fit_only = scores.data == nullptr;
    assert (data.dim1 == times.dim1);
    assert ((fit_only && scores.dim1 == 0) || data.dim1 == scores.dim1);
    
    std::vector<std::mutex> locks(fit_only ? 0 : std::min<std::size_t>(data.dim1, 10000));
    if (!fit_only)
        std::fill(scores.data, scores.data + data.dim1, 0);
    auto worker = [&](int ensemble_index) {
        auto& detector = ensemble[ensemble_index];
        for (int i = 0; i < data.dim1; i++) {
            FloatType score = detector.fitPredict(Vector<FloatType>(&data.data[i * data.dim2], data.dim2), times.data[i]);
            if (!fit_only && n_jobs < 2) {
                scores.data[i] += score / ensemble.size();
            }
            else if (!fit_only) {
                std::size_t lock_index = (data.dim1 < 10000) ? i : (boost::hash_value(i) % 10000);
                locks[lock_index].lock();
                scores.data[i] += score / ensemble.size();
                locks[lock_index].unlock();
            }
        }
    };
    fit_predict_ensemble<FloatType>(ensemble.size(), n_jobs, worker);
}


template<typename FloatType>
HSChains_wrapper<FloatType>::HSChains_wrapper(unsigned window, unsigned ensemble_size, int depth, int cms_w_param, int cms_d_param, int seed, unsigned n_jobs) :
    n_jobs(n_jobs)
{
    ensemble.reserve(ensemble_size);
    std::mt19937 rng(seed);
    for (unsigned i = 0; i < ensemble_size; i++)
        ensemble.emplace_back(window, depth, cms_w_param, cms_d_param, rng());
}

template<typename FloatType>
void HSChains_wrapper<FloatType>::fit(const NumpyArray2<FloatType> data) {
    fit_predict(data, {nullptr,0});
}

template<typename FloatType>
void HSChains_wrapper<FloatType>::fit_predict(const NumpyArray2<FloatType> data, NumpyArray1<FloatType> scores) {
    bool fit_only = scores.data == nullptr;
    assert (data.dim1 == times.dim1);
    assert ((fit_only && scores.dim1 == 0) || data.dim1 == scores.dim1);
    
    std::vector<std::mutex> locks(fit_only ? 0 : std::min<std::size_t>(data.dim1, 10000));
    if (!fit_only)
        std::fill(scores.data, scores.data + data.dim1, 0);
    auto worker = [&](int ensemble_index) {
        auto& detector = ensemble[ensemble_index];
        for (int i = 0; i < data.dim1; i++) {
            FloatType score = detector.fitPredict(Vector<FloatType>(&data.data[i * data.dim2], data.dim2));
            if (!fit_only && n_jobs < 2) {
                scores.data[i] += score;
            }
            else if (!fit_only) {
                std::size_t lock_index = (data.dim1 < 10000) ? i : (boost::hash_value(i) % 10000);
                locks[lock_index].lock();
                scores.data[i] += score;
                locks[lock_index].unlock();
            }
        }
    };
    fit_predict_ensemble<FloatType>(ensemble.size(), n_jobs, worker);
}

template<typename FloatType>
void HSChains_wrapper<FloatType>::set_initial_minmax(const NumpyArray1<FloatType> mins, const NumpyArray1<FloatType> maxs) {
    Vector<FloatType> mins_vec(mins.data, mins.dim1);
    Vector<FloatType> maxs_vec(maxs.data, maxs.dim1);
    for (auto& detector : ensemble) {
        detector.setInitialMinMax(mins_vec, maxs_vec);
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
template class HSTrees_wrapper<float>;
template class HSTrees_wrapper<double>;
template class HSChains_wrapper<float>;
template class HSChains_wrapper<double>;