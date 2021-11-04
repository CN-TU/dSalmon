// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

#include <algorithm>

#include "statisticstree_wrapper.h"


template<typename FloatType>
StatisticsTree_wrapper<FloatType>::StatisticsTree_wrapper(FloatType window) :
    window(window)
{ }

template<typename FloatType>
void StatisticsTree_wrapper<FloatType>::pruneExpired(FloatType now) {
    while (!tree.empty() && tree.front().second <= now) {
        tree.pop_front();
    }
}

template<typename FloatType>
void StatisticsTree_wrapper<FloatType>::fit_query(const NumpyArray2<FloatType> data, const NumpyArray1<FloatType> times, const NumpyArray1<unsigned> stats, const NumpyArray1<FloatType> quantiles, NumpyArray3<FloatType> result, NumpyArray1<long long> counts) {
    bool need_sum = false, need_min = false, need_max = false, need_median = false;
    for (int k = 0; k < stats.dim1; k++) {
        switch (stats.data[k]) {
            case STAT_SUM:
            case STAT_AVERAGE:
            case STAT_SQUARES_SUM:
            case STAT_VARIANCE:
                need_sum = true;
                break;
            case STAT_MIN:
                need_min = true;
                break;
            case STAT_MAX:
                need_max = true;
                break;
            case STAT_MEDIAN:
                need_median = true;
                break;
        }
    }
    for (int i = 0; i < data.dim1; i++) {
        pruneExpired(times.data[i]);
        tree.push_back(std::make_pair(Vector<FloatType>{&data.data[i * data.dim2], data.dim2}, times.data[i]+window));
        Vector<FloatType> sums, squares_sums, mins, maxs, medians;
        if (need_sum)
            tree.getStats(sums, squares_sums);
        if (need_min)
            mins = tree.getMins();
        if (need_max)
            maxs = tree.getMaxs();
        if (need_median)
            medians = tree.getMedians();
        std::size_t sw_size = tree.size();
        counts.data[i] = sw_size;

        for (int k = 0; k < stats.dim1; k++) {
            FloatType *destination = &result.data[i*result.dim2*result.dim3 + k*result.dim3];
            switch (stats.data[k]) {
                case STAT_SUM:
                    std::copy(sums.begin(), sums.end(), destination);
                    break;
                case STAT_AVERAGE:
                    for (int j = 0; j < data.dim2; j++)
                        destination[j] = sums[j] / sw_size;
                    break;
                case STAT_SQUARES_SUM:
                    std::copy(squares_sums.begin(), squares_sums.end(), destination);
                    break;
                case STAT_VARIANCE:
                    for (int j = 0; j < data.dim2; j++)
                        destination[j] = (squares_sums[j] - sums[j]*sums[j] / sw_size) / sw_size;
                    break;
                case STAT_MIN:
                    std::copy(mins.begin(), mins.end(), destination);
                    break;
                case STAT_MAX:
                    std::copy(maxs.begin(), maxs.end(), destination);
                    break;
                case STAT_MEDIAN:
                    std::copy(medians.begin(), medians.end(), destination);
                    break;
            }
        }
        for (int k = 0; k < quantiles.dim1; k++) {
            Vector<FloatType> q_results = tree.getQuantile(quantiles.data[k]);
            FloatType *destination = &result.data[i*result.dim2*result.dim3 + (stats.dim1+k)*result.dim3];
            std::copy(q_results.begin(), q_results.end(), destination);
        }
    }
}

template<typename FloatType>
void StatisticsTree_wrapper<FloatType>::transform_zscore(NumpyArray2<FloatType> data, const NumpyArray1<FloatType> times) {
    for (int i = 0; i < data.dim1; i++) {
        FloatType *row = &data.data[i*data.dim2];
        pruneExpired(times.data[i]);
        tree.push_back(std::make_pair(Vector<FloatType>{row, data.dim2}, times.data[i]+window));

        std::size_t sw_size = tree.size();
        Vector<FloatType> sums, squares_sums;
        tree.getStats(sums, squares_sums);
        
        for (int j = 0; j < data.dim2; j++) {
            FloatType mean = sums[j] / sw_size;
            FloatType stdev = std::sqrt( squares_sums[j] / sw_size - mean*mean );
            row[j] = (row[j] - mean) / stdev;
        }
    }
}

template<typename FloatType>
void StatisticsTree_wrapper<FloatType>::transform_quantile(NumpyArray2<FloatType> data, const NumpyArray1<FloatType> times, FloatType q) {
    for (int i = 0; i < data.dim1; i++) {
        FloatType *row = &data.data[i*data.dim2];
        pruneExpired(times.data[i]);
        tree.push_back(std::make_pair(Vector<FloatType>{row, data.dim2}, times.data[i]+window));

        Vector<FloatType> lower = tree.getQuantile(q);
        Vector<FloatType> upper = tree.getQuantile(1-q);
        for (int j = 0; j < data.dim2; j++) {
            row[j] = (row[j] - lower[j]) / (upper[j]-lower[j]);
        }
    }
}

template class StatisticsTree_wrapper<float>;
template class StatisticsTree_wrapper<double>;
