// Copyright (c) 2021 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

#ifndef DSALMON_STATISTICSTREE_H
#define DSALMON_STATISTICSTREE_H

#include <vector>
#include <list>
#include <limits>

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/ranked_index.hpp>
#include <boost/multi_index/identity.hpp>

#include "Vector.h"

template<typename T, typename FloatType=double>
class StatisticsTree {

    typedef std::pair<Vector<FloatType>, T> ValueType;

    typedef boost::multi_index_container<
        FloatType,
        boost::multi_index::indexed_by<boost::multi_index::ranked_non_unique<boost::multi_index::identity<FloatType>>>
    > TreeType;

    struct Point {
      ValueType value;
      std::vector<typename TreeType::template nth_index<0>::type::iterator> tree_iterators;
    };

    typedef std::list<Point> PointList;
    typedef typename PointList::iterator PointListIterator;

    struct Block {
        PointListIterator first_point;
        std::size_t contained_points;
        Vector<FloatType> sums;
        Vector<FloatType> squares_sums;
    };

    PointList points;
    std::vector<TreeType> trees;
    std::list<Block> blocks;

    const std::size_t MAX_BLOCK = 100;
  
    template<typename BaseIterator>
    class IteratorTempl : public BaseIterator {
      public:
        ValueType& operator*() const { return BaseIterator::operator*().value; }
        ValueType* operator->() const { return &BaseIterator::operator*().value; }
        IteratorTempl() {}
        IteratorTempl(BaseIterator it) : BaseIterator(it) {}
    };

  public:
    typedef IteratorTempl<typename PointList::iterator> iterator;
    typedef IteratorTempl<typename PointList::const_iterator> const_iterator;

    iterator begin() { return iterator(points.begin()); }
    const_iterator cbegin() const { return const_iterator(points.cbegin()); }
    iterator end() { return iterator(points.end()); }
    const_iterator cend() const { return const_iterator(points.cend()); }
    
    ValueType& front() { return points.front().value; }
    const ValueType& front() const { return points.front().value; }
    ValueType& back() { return points.back().value; }
    const ValueType& back() const { return points.back().value; }

    std::size_t size() const { return points.size(); }
    bool empty() const { return points.empty(); }

    void push_back(ValueType&& value) {
        Vector<FloatType>& data = value.first;
        std::size_t dimension = data.size();

        points.emplace_back();
        Point &new_entry = points.back();

        if (trees.empty())
            trees.resize(dimension);
        new_entry.tree_iterators.resize(dimension);
        for (std::size_t i = 0; i < trees.size(); i++) {
            auto inserted = trees[i].template get<0>().insert(data[i]);
            new_entry.tree_iterators[i] = inserted.first;
        }

        if (blocks.empty() || blocks.back().contained_points >= MAX_BLOCK) {
            blocks.emplace_back();
            Block& new_block = blocks.back();
            new_block.contained_points = 0;
            new_block.first_point = std::prev(points.end());
            new_block.sums.resize(dimension);
            new_block.squares_sums.resize(dimension);
        }
        Block& last_block = blocks.back();
        last_block.contained_points++;
        last_block.sums += data;
        for (std::size_t i = 0; i < dimension; i++)
            last_block.squares_sums[i] += data[i] * data[i];

        new_entry.value = std::move(value);
    }

    void pop_front() {
        auto& iterators = points.front().tree_iterators;
        for (std::size_t i = 0; i < iterators.size(); i++)
            trees[i].template get<0>().erase(iterators[i]);
        if (!blocks.empty() && blocks.front().first_point == points.begin())
            blocks.pop_front();
        points.pop_front();
    }


    void getStats(Vector<FloatType>& sums, Vector<FloatType>& squares_sums) {
        sums.clear();
        squares_sums.clear();
        sums.resize(trees.size());
        squares_sums.resize(trees.size());

        PointListIterator sum_to = blocks.empty() ? points.end() : blocks.front().first_point;
        for (PointListIterator it = points.begin(); it != sum_to; it++) {
            Vector<FloatType>& data = it->value.first;
            sums += data;
            for(std::size_t i = 0; i < data.size(); i++)
                squares_sums[i] += data[i] * data[i];
        }
        for (auto& block : blocks) {
            sums += block.sums;
            squares_sums += block.squares_sums;
        }
    }

    Vector<FloatType> getMins() { return getQuantile(0.0); }
    Vector<FloatType> getMaxs() { return getQuantile(1.0); }
    Vector<FloatType> getMedians() { return getQuantile(0.5); }

    Vector<FloatType> getQuantile(FloatType q) {
        Vector<FloatType> result;
        if (points.empty()) {
            result.resize(trees.size(), std::numeric_limits<FloatType>::quiet_NaN());
            return result;
        }
        result.resize(trees.size());
        if (q <= 0.0) {
            for (std::size_t i = 0; i < trees.size(); i++) {  
                auto& tree_index = trees[i].template get<0>();
                result[i] = *tree_index.begin();
            }
        }
        else if (q >= 1.0) {
            for (std::size_t i = 0; i < trees.size(); i++) {  
                auto& tree_index = trees[i].template get<0>();
                result[i] = *tree_index.rbegin();
            }
        }
        else {
            std::size_t max_n = points.size()-1;
            for (std::size_t i = 0; i < trees.size(); i++) {  
                auto& tree_index = trees[i].template get<0>();
                result[i] = *(tree_index.nth(static_cast<std::size_t>(std::floor(max_n * q))));
            }
        }
        return result;
    }
};

#endif