// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

#ifndef DSALMON_MTREE_WRAPPER_H
#define DSALMON_MTREE_WRAPPER_H

#include "MTree.h"
#include "Vector.h"

#include "array_types.h"
#include "distance_wrappers.h"

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/ranked_index.hpp>
#include <boost/multi_index/member.hpp>

template<typename FloatType>
class MTreeSelector_wrapper;
template<typename FloatType>
class MTreeRangeQuery_wrapper;
template<typename FloatType>
class MTreeKnnQuery_wrapper;

template<typename FloatType>
class MTree_wrapper {
    typedef Vector<FloatType> VectorType;

    int dimension;
    int counter;
    unsigned insert_jobs, query_jobs;

    typedef MTree<VectorType,long long,FloatType> Tree;
    typedef typename Tree::iterator TreeIterator;

    struct Entry {
        long long key;
        TreeIterator it;
    };
    typedef boost::multi_index_container<
        Entry,
        boost::multi_index::indexed_by<boost::multi_index::ranked_unique<boost::multi_index::member<Entry,long long,&Entry::key>>>
    > EntryMap;
    typedef typename EntryMap::template nth_index<0>::type EntryIndex;
    typedef typename EntryIndex::iterator EntryIndexIterator;
    
    Tree tree;
    EntryMap entries;

  public:
    MTree_wrapper(Distance_wrapper<FloatType>* distance, int min_node_size, int max_node_size, int split_sampling, unsigned insert_jobs, unsigned query_jobs);

    void insert(const NumpyArray2<FloatType> data, const NumpyArray1<long long> indices);
    void remove(MTreeSelector_wrapper<FloatType> *selector);
    
    void update(const NumpyArray1<long long> keys, const NumpyArray2<FloatType> data);
    void updateByIndex(const NumpyArray1<long long> keys, const NumpyArray2<FloatType> data);
    void updateByIndexSlice(long long from, long long to, long long step, const NumpyArray2<FloatType> data);
    
    void keys(MTreeSelector_wrapper<FloatType> *selector, NumpyArray1<long long> keys);
    void indices(MTreeSelector_wrapper<FloatType> *selector, NumpyArray1<long long> indices);
    void getPoints(MTreeSelector_wrapper<FloatType> *selector, NumpyArray2<FloatType> data);

    int size() { return entries.size(); }
    int dataDimension() { return dimension; }
    void clear();

    std::string serialize();
    bool unserialize(std::string data);
    
    friend class MTreeSelector_wrapper<FloatType>;
    friend class MTreeRangeQuery_wrapper<FloatType>;
    friend class MTreeKnnQuery_wrapper<FloatType>;
};
DEFINE_FLOATINSTANTIATIONS(MTree)

template<typename FloatType>
class MTreeSelector_wrapper {
    typedef typename MTree_wrapper<FloatType>::EntryIndex::iterator MapIterator;
    typedef std::vector<MapIterator> IteratorList;
    MTree_wrapper<FloatType> *tree;
    bool all_found;
    IteratorList entries;
    
    IteratorList getIteratorsAndClear() { return std::move(entries); }
    void sort();
  public:
    MTreeSelector_wrapper(MTree_wrapper<FloatType> *tree, const NumpyArray1<long long> keys, bool by_index);
    MTreeSelector_wrapper(MTree_wrapper<FloatType> *tree, long long from, long long to, long long step);
    unsigned char allOk() { return all_found; }
    void getFoundMask(NumpyArray1<unsigned char> found);
    int size() { return entries.size(); }
    
    friend class MTree_wrapper<FloatType>;
};
DEFINE_FLOATINSTANTIATIONS(MTreeSelector)



template<typename FloatType>
class MTreeRangeQuery_wrapper {
    std::vector<std::vector<std::pair<typename MTree_wrapper<FloatType>::TreeIterator,FloatType>>> results;
    std::vector<int> result_lengths;
    
  public:
    MTreeRangeQuery_wrapper(MTree_wrapper<FloatType>* tree, const NumpyArray2<FloatType> data, FloatType radius);
    void result(NumpyArray1<long long> indices, NumpyArray1<FloatType> distances);
    void resultLengths(NumpyArray1<int> lengths);
    int resultTotalLength();
};
DEFINE_FLOATINSTANTIATIONS(MTreeRangeQuery)

template<typename FloatType>
class MTreeKnnQuery_wrapper {
    std::vector<std::vector<std::pair<typename MTree_wrapper<FloatType>::TreeIterator,FloatType>>> results;
    std::vector<long long> result_lengths;
    
  public:
    MTreeKnnQuery_wrapper(MTree_wrapper<FloatType>* tree, const NumpyArray2<FloatType> data, int k, bool sort, FloatType min_radius, FloatType max_radius, bool reverse, bool extend_for_ties);
    void result(NumpyArray1<long long> indices, NumpyArray1<FloatType> distances);
    void resultLengths(NumpyArray1<int> lengths);
    int resultTotalLength();
};
DEFINE_FLOATINSTANTIATIONS(MTreeKnnQuery)


#endif
