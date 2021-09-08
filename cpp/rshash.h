// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

#ifndef DSALMON_RSHASH_H
#define DSALMON_RSHASH_H

#include <deque>
#include <boost/functional/hash.hpp>
#include <cmath>

#include "Vector.h"

template<typename FloatType>
class RSHash {

    FloatType window;
    FloatType s_param;
    FloatType f_param;
    std::vector<FloatType> alpha;
    int r_param;
    int dimension;
    int cms_w_param;
    int cms_d_param;
    std::vector<std::uint64_t> cms_table;

    std::mt19937 rng;

    std::vector<bool> dimension_selected;

    // represents one stored data point in the sliding window
    struct Sample {
        std::shared_ptr<Vector<FloatType>> data;
        FloatType expire_time;
    };

    typedef std::deque<Sample> SlidingWindow;
    typedef typename SlidingWindow::iterator SlidingWindowIterator;
    SlidingWindow sliding_window;

    void selectDimensions(int dimensions_to_draw) {
        dimension_selected.clear();
        if (dimensions_to_draw == dimension) {
            dimension_selected.resize(dimension, true);
        }
        else {
            dimension_selected.resize(dimension, false);
            std::vector<int> available(dimension);
            for (int i = 0; i < dimension; i++)
                available[i] = i;
            for (; dimensions_to_draw > 0; dimensions_to_draw--) {
                int r = std::uniform_int_distribution<int>{0,(int)available.size()-1}(rng);
                dimension_selected[available[r]] = true;
                available[r] = available.back();
                available.pop_back();
            }
        }
    }

    std::vector<int> computeCmsBins(const Vector<FloatType>& data) {
        std::int64_t transformed_sample[dimension];
        for (int i = 0; i < dimension; i++) {
            if (dimension_selected[i]) {
                if (!std::isinf(data[i])) {
                    transformed_sample[i] = (data[i] + alpha[i])/f_param;
                }
                else if (data[i] > 0) {
                    // positive infinity
                    transformed_sample[i] = std::numeric_limits<std::int64_t>::max();
                }
                else {
                    // negative infinity
                    transformed_sample[i] = std::numeric_limits<std::int64_t>::min();
                }
            }
            else {
                transformed_sample[i] = -1;
            }
        }
        std::vector<int> bins(cms_w_param);
        for (int i = 0; i < cms_w_param; i++) {
            std::size_t seed = boost::hash_value(i);
            for (std::int64_t y : transformed_sample)
                boost::hash_combine(seed, y);
            bins[i] = seed % cms_d_param;
        }
        return bins;
    }

    void removeFromCms(const Vector<FloatType>& data) {
        std::vector<int> cms_bins = computeCmsBins(data);
        for (int i = 0; i < cms_w_param; i++) {
            cms_table[cms_bins[i] + i*cms_d_param]--;
        }
    }

    void pruneExpired(FloatType now) {
        while (!sliding_window.empty() && sliding_window.front().expire_time <= now) {
            removeFromCms(*(sliding_window.front().data));
            sliding_window.pop_front();
        }
    }

    void initialize(int dimension) {
        this->dimension = dimension;
        alpha.resize(dimension);
        for (int i = 0; i < dimension; i++) {
            alpha[i] = std::uniform_real_distribution<FloatType>{0, f_param}(rng);
        }

        FloatType log_basis = std::max<FloatType>(2,1/f_param);
        FloatType pre_r = std::log(s_param) / std::log(log_basis);
        // We have to be sure that r_min <= r_max. This holds if s >= 4
        int r_min = std::round(1+pre_r/2);
        int r_max = std::round(pre_r);
        int dimensions_to_draw = std::uniform_int_distribution<int>{r_min,r_max}(rng);
        if (dimensions_to_draw > dimension)
            dimensions_to_draw = dimension;
        selectDimensions(dimensions_to_draw);
    }

  public:
    RSHash(FloatType window, FloatType s_param, unsigned cms_w_param, unsigned cms_d_param, int seed) :
        window(window),
        s_param(s_param),
        dimension(-1),
        cms_w_param(cms_w_param),
        cms_d_param(cms_d_param),
        rng(seed)
    {
        assert (s_param >= 4);
        FloatType sqrt_s = std::sqrt(s_param);
        f_param = std::uniform_real_distribution<FloatType>{1/sqrt_s,1-1/sqrt_s}(rng);
        cms_table.resize(cms_w_param * cms_d_param);
    }

    FloatType fitPredict(std::shared_ptr<Vector<FloatType>> data, FloatType now) {
        if (dimension == -1) {
            // Lazy-initialize when the first sample is seen. Before that,
            // we don't know the dimension.
            initialize(data->size());
        }
        pruneExpired(now);

        std::vector<int> cms_bins = computeCmsBins(*data);
        std::uint64_t event_count = std::numeric_limits<std::uint64_t>::max();
        for (int i = 0; i < cms_w_param; i++) {
            std::uint64_t& cms_entry = cms_table[cms_bins[i] + i*cms_d_param];
            event_count = std::min<std::uint64_t>(event_count, cms_entry);
            cms_entry++;
        }
        sliding_window.push_back({data, now + window});
        // RS-Hash reports a score for normality. To be consistent with remaining
        // algorithms, we multiply with -1 to return an outlier score.
        return -std::log2(event_count+1);
    }

    void fit(std::shared_ptr<Vector<FloatType>> data, FloatType now) {
        // Just for consistency. We cannot make this fast than fitPredict().
        fitPredict(data, now);
    }

    std::size_t windowSize() {
        return sliding_window.size();
    }

    // interface for investigating the contents of the SW
    class WindowSample{
        SlidingWindowIterator it;
    public:
        WindowSample(SlidingWindowIterator it) : it(it) {}
        Vector<FloatType> getSample() { return *it->data; }
        FloatType getExpireTime() { return it->expire_time; }
    };
    
    class iterator : public SlidingWindowIterator {
      public:
        WindowSample operator*() { return WindowSample(static_cast<SlidingWindowIterator>(*this)); };
        iterator() {}
        iterator(SlidingWindowIterator it) : SlidingWindowIterator(it) {}
    };
    
    iterator begin() { return iterator(sliding_window.begin()); }
    iterator end() { return iterator(sliding_window.end()); }
};

#endif
