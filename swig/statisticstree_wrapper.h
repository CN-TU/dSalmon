// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

#ifndef DSALMON_STATISTICSTREE_WRAPPER_H
#define DSALMON_STATISTICSTREE_WRAPPER_H

#include "statisticsTree.h"
#include "array_types.h"


template<typename FloatType=double>
class StatisticsTree_wrapper {

    StatisticsTree<FloatType,FloatType> tree;
    FloatType window;

    void pruneExpired(FloatType now);

  public:
    enum {
        STAT_SUM = 1,
        STAT_AVERAGE,
        STAT_SQUARES_SUM,
        STAT_VARIANCE,
        STAT_MIN,
        STAT_MAX,
        STAT_MEDIAN
    };
    StatisticsTree_wrapper(FloatType window);
    void fit_query(const NumpyArray2<FloatType> data, const NumpyArray1<FloatType> times, const NumpyArray1<unsigned> stats, const NumpyArray1<FloatType> quantiles, NumpyArray3<FloatType> result, NumpyArray1<long long> counts);
    void transform_zscore(NumpyArray2<FloatType> data, const NumpyArray1<FloatType> times);
    void transform_quantile(NumpyArray2<FloatType> data, const NumpyArray1<FloatType> times, FloatType q);
};
DEFINE_FLOATINSTANTIATIONS(StatisticsTree)

#endif
