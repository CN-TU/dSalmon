// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

#ifndef DSALMON_XSTREAM_H
#define DSALMON_XSTREAM_H

#include <cmath>
#include <limits>
#include <utility>
#include <random>
#include <vector>

#include <boost/functional/hash.hpp>

#include "Vector.h"

template<typename FloatType=double>
class HSChain {
    // algorithm parameters
    std::uint64_t window_param;
    int depth;
    int cms_w_param;
    int cms_d_param;
    std::mt19937 rng;

    // values randomly initialized at the first point
    std::vector<FloatType> random_offset;
    std::vector<std::pair<int,FloatType>> dimensions;

    // model and reference model
    std::uint64_t points_in_current_window;
    bool reference_window_valid;
    std::vector<FloatType> min_values;
    std::vector<FloatType> max_values;
    std::vector<std::vector<std::uint64_t>> cms_tables;
    std::vector<FloatType> min_values_reference;
    std::vector<FloatType> max_values_reference;
    std::vector<std::vector<std::uint64_t>> cms_reference_tables;

    void computeCmsBins(const std::vector<std::uint64_t>& xstream_bins, std::vector<int>& cms_bins) {
        cms_bins.resize(cms_w_param);
        for (int i = 0; i < cms_w_param; i++) {
            std::size_t seed = boost::hash_value(i);
            for (std::uint64_t y : xstream_bins)
                boost::hash_combine(seed, y);
            cms_bins[i] = seed % cms_d_param;
        }
    }

    void initialize(int dimension) {
        std::uniform_int_distribution<int> dim_distribution{0,dimension-1};
        std::uniform_real_distribution<FloatType> s_distribution{0, 0.5};
        std::vector<int> dimension_multiplicity(dimension);
        for (int i = 0; i < depth; i++) {
            int dim = dim_distribution(rng);
            dimension_multiplicity[dim]++;
            FloatType bin_count_for_dimension = std::exp2(static_cast<FloatType>(dimension_multiplicity[dim]));
            dimensions.emplace_back(dim, bin_count_for_dimension);
        }
        for (int i = 0; i < dimension; i++) {
            random_offset.push_back(s_distribution(rng));
        }
        cms_tables.resize(depth);
        cms_reference_tables.resize(depth);
        for (auto& cms_table : cms_tables)
            cms_table.resize(cms_w_param * cms_d_param);
        for (auto& cms_reference_table : cms_reference_tables)
            cms_reference_table.resize(cms_w_param * cms_d_param);
        min_values.resize(dimension, std::numeric_limits<FloatType>::infinity());
        max_values.resize(dimension, -std::numeric_limits<FloatType>::infinity());
        // TODO: maybe implement mode to estimate this from first initial window
        min_values_reference.resize(dimension, 0);
        max_values_reference.resize(dimension, 1);
        points_in_current_window = 0;
        reference_window_valid = false;
    }

  public:
    HSChain(std::uint64_t window, int depth, int cms_w_param, int cms_d_param, int seed) :
        window_param(window),
        depth(depth),
        cms_w_param(cms_w_param),
        cms_d_param(cms_d_param),
        rng(seed)
    { }

    void setInitialMinMax(const Vector<FloatType>& mins, const Vector<FloatType>& maxs) {
        if (dimensions.empty())
            initialize(mins.size());
        min_values_reference = mins;
        max_values_reference = maxs;
    }

    FloatType fitPredict(const Vector<FloatType>& data) {
        if (dimensions.empty())
            initialize(data.size());
        Vector<FloatType> data_normalized(data.size());

        for (std::size_t i = 0; i < data.size(); i++) {
            min_values[i] = std::min(min_values[i], data[i]);
            max_values[i] = std::max(min_values[i], data[i]);
            // TODO: what if max_values_reference[i] == min_values_reference[i]
            data_normalized[i] = (data[i]-min_values_reference[i]) / (max_values_reference[i]-min_values_reference[i]);
        }

        FloatType score = std::numeric_limits<FloatType>::infinity();
        std::vector<std::uint64_t> xstream_bins(data.size());
        std::vector<int> cms_bins(cms_w_param);
        for (int l = 0; l < depth; l++) {
            int dim;
            FloatType bin_count_for_dimension;
            std::tie(dim, bin_count_for_dimension) = dimensions[l];
            xstream_bins[dim] =  // TODO: should handle nan
                std::max<FloatType>(0, std::min<FloatType>(bin_count_for_dimension-1, data_normalized[dim]*bin_count_for_dimension + random_offset[dim]));
            computeCmsBins(xstream_bins, cms_bins);
            std::vector<std::uint64_t>& cms_table = cms_tables[l];
            std::vector<std::uint64_t>& cms_reference_table = cms_reference_tables[l];
            std::uint64_t event_count = std::numeric_limits<std::uint64_t>::max();
            for (int i = 0; i < cms_w_param; i++) {
                int index = cms_bins[i] + i*cms_d_param;
                cms_table[index]++;
                event_count = std::min<std::uint64_t>(event_count, cms_reference_table[index]);
            }
            score = std::min<FloatType>(score, event_count * std::exp2(static_cast<FloatType>(l+1)) );
        }
        if (++points_in_current_window == window_param) {
            std::swap(cms_tables, cms_reference_tables);
            std::swap(min_values, min_values_reference);
            std::swap(max_values, max_values_reference);
            for (auto& cms_table : cms_tables) {
                std::fill(cms_table.begin(), cms_table.end(), 0);
            }
            std::fill(min_values.begin(), min_values.end(), std::numeric_limits<FloatType>::infinity());
            std::fill(max_values.begin(), max_values.end(), -std::numeric_limits<FloatType>::infinity());
            points_in_current_window = 0;
            reference_window_valid = true;
        }
        // xStream reports a score for normality. To be consistent with remaining
        // algorithms, we multiply with -1 to return an outlier score.
        return reference_window_valid ? -score : std::numeric_limits<FloatType>::quiet_NaN();
    }
};

#endif
