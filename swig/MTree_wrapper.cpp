// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

#include <assert.h>
#include <map>

#include <thread>

#include <iostream> // TODO
#include <atomic>
#include <sstream>

#include "MTree_wrapper.h"

template<typename FloatType>
MTree_wrapper<FloatType>::MTree_wrapper(Distance_wrapper<FloatType>* distance, int min_node_size, int max_node_size, int split_sampling, unsigned insert_jobs, unsigned query_jobs) :
    dimension(-1), counter(0), insert_jobs(insert_jobs), query_jobs(query_jobs ? query_jobs : 1),
    tree(distance->getFunction(), min_node_size, max_node_size, split_sampling)
{
    
}

template<typename FloatType>
void MTree_wrapper<FloatType>::insert(const NumpyArray2<FloatType> data, const NumpyArray1<long long> indices) {
    assert(dimension == -1 || dimension == data.dim2);
    assert(data.dim1 == indices.dim1);
    dimension = data.dim2;
    EntryIndex& index = entries.template get<0>();
    const int last_existing = index.empty() ? -1 : std::prev(index.end())->key;
    
    std::vector<TreeIterator> tree_entries;
    std::vector<EntryIndexIterator> index_entries;
    for (int i = 0; i < data.dim1; i++, counter++) {
        while (counter <= last_existing && index.find(counter) != index.end())
            counter++;
        indices.data[i] = counter;
    }
    tree_entries.reserve(data.dim1);
    index_entries.reserve(data.dim1);
    std::atomic<int> global_i(0);
    auto worker = [this, &data, &indices, &global_i, &tree_entries]() {
        for (int i = global_i++; i < data.dim1; i = global_i++) {
            FloatType *vec_data = &data.data[i * data.dim2];
            tree_entries[i] = tree.insert(
                tree.end(),
                // TODO: use std::piecewise_construct_t
                std::make_pair(VectorType{vec_data, data.dim2}, indices.data[i])
            );
        }
    };
    int n_jobs = std::min<int>(data.dim1, insert_jobs);
    // TODO: use a different implementation when n_jobs==1
    std::thread threads[n_jobs];
    for (int i = 0; i < n_jobs; i++)
        threads[i] = std::thread{worker};
    for (int i = 0; i < data.dim1; i++) {
        Entry new_entry;
        new_entry.key = indices.data[i];
        std::tie(index_entries[i], std::ignore) = index.insert(new_entry);
    }
    for (int i = 0; i < n_jobs; i++)
        threads[i].join();

    for (int i = 0; i < data.dim1; i++) {
        class ItUpdater {
            TreeIterator new_it;
          public:
            ItUpdater(TreeIterator new_it) : new_it(new_it) {}
            void operator()(Entry& entry) {
                entry.it = new_it;
            }
        };
        index.modify(index_entries[i], ItUpdater{tree_entries[i]});
    }
}

template<typename FloatType>
void MTree_wrapper<FloatType>::remove(MTreeSelector_wrapper<FloatType> *selector) {
    EntryIndex& index = entries.template get<0>();
    EntryIndexIterator last = index.end();
    selector->sort();
    for (auto it : selector->getIteratorsAndClear()) {
        if (it != last) {
            tree.erase(it->it);
            index.erase(it);
        }
        last = it;
    }
    if (entries.empty()) {
        dimension = -1;
    }
}

namespace {
    template<typename FloatType>
    class VectorUpdater {
        FloatType *new_data;
    public:
        VectorUpdater(FloatType *new_data) : new_data(new_data) {}
        void operator() (Vector<FloatType>& vector, long long&) {
            int i = 0;
            for (FloatType& element : vector) {
                element = new_data[i];
                i++;
            }
        }
    };
}

template<typename FloatType>
void MTree_wrapper<FloatType>::update(const NumpyArray1<long long> keys, const NumpyArray2<FloatType> data) {
    // unlike updateByIndex and updateByIndexSlice, new entries might be inserted in this function
    assert(dimension == -1 || dimension == data.dim2);
    assert(keys.dim1 == data.dim1);
    dimension = data.dim2;
    EntryIndex& index = entries.template get<0>();
    for (int i = 0; i < keys.dim1; i++) {
        EntryIndexIterator it = index.lower_bound(keys.data[i]);
        if (it->key == keys.data[i]) {
            tree.modify(it->it, VectorUpdater<FloatType>{&data.data[i * data.dim2]});
        }
        else {
            Entry new_entry;
            tree.push_back(std::make_pair(VectorType{&data.data[i * data.dim2], data.dim2}, keys.data[i]));
            new_entry.key = keys.data[i];
            new_entry.it = std::prev(tree.end());
            index.emplace_hint(it, new_entry);
        }
    }
}

template<typename FloatType>
void MTree_wrapper<FloatType>::updateByIndex(const NumpyArray1<long long> keys, const NumpyArray2<FloatType> data) {
    assert(keys.dim1 == data.dim1);
    EntryIndex& index = entries.template get<0>();
    for (int i = 0; i < keys.dim1; i++) {
        EntryIndexIterator it = index.nth(keys.data[i]);
        tree.modify(it->it, VectorUpdater<FloatType>{&data.data[i * data.dim2]});
    }
}

template<typename FloatType>
void MTree_wrapper<FloatType>::updateByIndexSlice(long long from, long long to, long long step, const NumpyArray2<FloatType> data) {
    assert(dimension == data.dim2);
    assert(from >= 0 && from < entries.size());
    assert(to >= -1 && to <= entries.size());
    assert(step != 0);
    assert((to < from) == (step < 0));
    assert((to-from)/step == data.dim1);
    EntryIndex& index = entries.template get<0>();
    if (step < 10) {
        EntryIndexIterator it = index.nth(from);
        for (int i = 0; i < data.dim1; i++) {
            tree.modify(it->it, VectorUpdater<FloatType>{&data.data[i * data.dim2]});
            std::advance(it, step);
        }
    }
    else {
        int ind = from;
        for (int i = 0; i < data.dim1; i++, ind += step) {
            EntryIndexIterator it = index.nth(ind);
            tree.modify(it->it, VectorUpdater<FloatType>{&data.data[i * data.dim2]});
        }
    }
}

template<typename FloatType>
void MTree_wrapper<FloatType>::keys(MTreeSelector_wrapper<FloatType> *selector, NumpyArray1<long long> keys) {
    assert(selector->size() == keys.dim1);
    int i = 0;
    for (auto it : selector->getIteratorsAndClear()) {
        keys.data[i] = it->key;
        i++;
    }
}

template<typename FloatType>
void MTree_wrapper<FloatType>::indices(MTreeSelector_wrapper<FloatType> *selector, NumpyArray1<long long> indices) {
    assert(selector->size() == indices.dim1);
    EntryIndex& index = entries.template get<0>();
    int i = 0;
    for (auto it : selector->getIteratorsAndClear()) {
        indices.data[i] = index.rank(it);
        i++;
    }
}

template<typename FloatType>
void MTree_wrapper<FloatType>::getPoints(MTreeSelector_wrapper<FloatType> *selector, NumpyArray2<FloatType> data) {
    assert(data.dim2 == dimension);
    assert(selector->size() == data.dim1);
    int i = 0;
    for (auto it : selector->getIteratorsAndClear()) {
        const VectorType& vec_data = it->it->first;
        std::copy(vec_data.cbegin(), vec_data.cend(), &data.data[data.dim2 * i]);
        i++;
    }
}

static const int SERIALIZATION_VERSION = 1;

template<typename FloatType>
std::string MTree_wrapper<FloatType>::serialize() {
    auto key_serializer = [](std::ostream& os, const VectorType& vec) {
        vec.serialize(os);
    };
    auto int_serializer = [](std::ostream& os, int i) {
        serializeInt<std::uint64_t>(os, i);
    };
    std::stringstream result;
    serializeInt<std::uint64_t>(result, SERIALIZATION_VERSION);
    serializeInt<std::int64_t>(result, dimension);
    serializeInt<std::uint64_t>(result, counter);
    serializeInt<std::uint64_t>(result, insert_jobs);
    serializeInt<std::uint64_t>(result, query_jobs);
    tree.serialize(result, key_serializer, int_serializer);
    return result.str();
}

template<typename FloatType>
bool MTree_wrapper<FloatType>::unserialize(std::string data) {
    std::stringstream stream{data};
    auto key_unserializer = [](std::istream& is) {
        VectorType vec;
        vec.unserialize(is);
        return vec;
    };
    auto int_unserializer = [](std::istream& is) {
        return unserializeInt<std::uint64_t>(is);
    };
    int version = unserializeInt<std::uint64_t>(stream);
    if (version != SERIALIZATION_VERSION)
        return false;
    dimension = unserializeInt<std::int64_t>(stream);
    counter = unserializeInt<std::uint64_t>(stream);
    insert_jobs = unserializeInt<std::uint64_t>(stream);
    query_jobs = unserializeInt<std::uint64_t>(stream);
    tree.unserialize(stream, key_unserializer, int_unserializer);
    for (auto it = tree.begin(); it != tree.end(); it++) {
        Entry new_entry;
        new_entry.it = it;
        new_entry.key = it->second;
        entries.template get<0>().insert(new_entry);
    }
    return true;
}

template<typename FloatType>
void MTree_wrapper<FloatType>::clear() {
    entries.clear();
    tree.clear();
    dimension = -1;
}


template<typename FloatType>
MTreeSelector_wrapper<FloatType>::MTreeSelector_wrapper(MTree_wrapper<FloatType> *tree, const NumpyArray1<long long> keys, bool by_index)
    : tree(tree)
{
    entries.reserve(keys.dim1);
    auto& index = tree->entries.template get<0>();
    all_found = true;
    for (int i = 0; i < keys.dim1; i++) {
        auto it = by_index ?
            index.nth(keys.data[i]) :
            index.find(keys.data[i]);
        entries.push_back(it);
        all_found &= (entries.back() != index.end());
    }
}

template<typename FloatType>
void MTreeSelector_wrapper<FloatType>::getFoundMask(NumpyArray1<unsigned char> found) {
    assert(entries.size() == found.dim1);
    auto& index = tree->entries.template get<0>();
    int i = 0;
    for (auto it : entries) {
        found.data[i] = (it != index.end());
        i++;
    }
}

template<typename FloatType>
MTreeSelector_wrapper<FloatType>::MTreeSelector_wrapper(MTree_wrapper<FloatType> *tree, long long from, long long to, long long step)
    : tree(tree)
{
    assert(from >= 0 && from < tree->entries.size());
    assert(to >= -1 && to <= tree->entries.size());
    assert(step != 0);
    assert((to < from) == (step < 0));
    entries.reserve((to-from)/step);
    auto& index = tree->entries.template get<0>();
    if (step < 10) {
        auto it = index.nth(from);
        // step*i < step*to <=> (step>0) ? (i < to) : (i > to)
        for (int i = from; step*i < step*to; i += step) {
            entries.push_back(it);
            std::advance(it, step);
        }
    }
    else {
        for (int i = from; step*i < step*to; i += step) {
            entries.push_back(index.nth(i));
        }
    }
    all_found = true;
}

template<typename FloatType>
void MTreeSelector_wrapper<FloatType>::sort() {
    // we only need sorting functionality to filter out duplicates,
    // so sorting by address is sufficient
    struct {
        bool operator() (MapIterator a, MapIterator b) {
            return &*a < &*b;
        }
    } sorter;
    std::sort(entries.begin(), entries.end(), sorter);
}


template<typename FloatType>
MTreeRangeQuery_wrapper<FloatType>::MTreeRangeQuery_wrapper(MTree_wrapper<FloatType>* tree, const NumpyArray2<FloatType> data, FloatType radius) {
    assert(data.dim2 == tree->dimension);
    results.resize(data.dim1);
    result_lengths.resize(data.dim1);
    std::atomic<int> global_i{0};
    auto worker = [this, tree, &data, radius, &global_i]() {
        for (int i = global_i++; i < data.dim1; i = global_i++) {
            auto query = tree->tree.rangeSearch(Vector<FloatType>{&data.data[i * data.dim2], data.dim2}, radius);
            for(; !query.atEnd(); ++query)
                results[i].push_back(*query);
            result_lengths[i] = results[i].size();
        }
    };
    int n_jobs = std::min<int>(data.dim1, tree->query_jobs);
    if (n_jobs < 2) {
        worker();
    }
    else {
        std::thread threads[n_jobs];
        for (int i = 0; i < n_jobs; i++)
            threads[i] = std::thread(worker);
        for (int i = 0; i < n_jobs; i++)
            threads[i].join();
    }
}

template<typename FloatType>
void MTreeRangeQuery_wrapper<FloatType>::result(NumpyArray1<long long> indices, NumpyArray1<FloatType> distances) {
    assert(indices.dim1 == distances.dim1);
    int i = 0;
    for (auto& result : results) {
        for (auto& item : result) {
            assert(i < data.dim1);
            auto it = item.first;
            indices.data[i] = it->second;
            distances.data[i] = item.second;
            i++;
        }
    }
    assert(i == data.dim1);
}

template<typename FloatType>
void MTreeRangeQuery_wrapper<FloatType>::resultLengths(NumpyArray1<int> lengths) {
    assert(lengths.dim1 == result_lengths.size());
    std::copy(result_lengths.begin(), result_lengths.end(), lengths.data);
}

template<typename FloatType>
int MTreeRangeQuery_wrapper<FloatType>::resultTotalLength() {
    int total_length = 0;
    for (auto& result : results)
        total_length += result.size();
    return total_length;
}


template<typename FloatType>
MTreeKnnQuery_wrapper<FloatType>::MTreeKnnQuery_wrapper(MTree_wrapper<FloatType>* tree, const NumpyArray2<FloatType> data, int k, bool sort, FloatType min_radius, FloatType max_radius, bool reverse, bool extend_for_ties) {
    assert(data.dim2 == tree->dimension);
    results.resize(data.dim1);
    result_lengths.resize(data.dim1);
    typedef typename MTree_wrapper<FloatType>::Tree::ValueType TreeValueType;
    auto tie_breaker = [](const TreeValueType& a, const TreeValueType& b) {
        return a.second < b.second;
    };
    std::atomic<int> global_i{0};
    auto worker = [this, tree, &data, &global_i, k, sort, min_radius, max_radius, reverse, extend_for_ties, &tie_breaker]() {
        for (int i = global_i++; i < data.dim1; i = global_i++) {
            results[i] = tree->tree.knnSearch(Vector<FloatType>{&data.data[i*data.dim2], data.dim2}, k, sort, min_radius, max_radius, reverse, extend_for_ties, tie_breaker);
            result_lengths[i] = results[i].size();
        }
    };
    // TODO: use a different implementation when n_jobs==1
    int n_jobs = std::min<int>(data.dim1, tree->query_jobs);
    if (n_jobs < 2) {
        worker();
    }
    else {
        std::thread threads[n_jobs];
        for (int i = 0; i < n_jobs; i++)
            threads[i] = std::thread(worker);
        for (int i = 0; i < n_jobs; i++)
            threads[i].join();
    }
}

template<typename FloatType>
void MTreeKnnQuery_wrapper<FloatType>::result(NumpyArray1<long long> indices, NumpyArray1<FloatType> distances) {
    assert(data.dim1 == distances.dim1);
    int i = 0;
    for (auto& result : results) {
        for (auto& item : result) {
            assert(i < data.dim1);
            auto it = item.first;
            indices.data[i] = it->second;
            distances.data[i] = item.second;
            i++;
        }
    }
    assert(i == data.dim1);
}

template<typename FloatType>
void MTreeKnnQuery_wrapper<FloatType>::resultLengths(NumpyArray1<int> lengths) {
    assert(lengths.dim1 == result_lengths.size());
    std::copy(result_lengths.begin(), result_lengths.end(), lengths.data);
}

template<typename FloatType>
int MTreeKnnQuery_wrapper<FloatType>::resultTotalLength() {
    int total_length = 0;
    for (auto& result : results)
        total_length += result.size();
    return total_length;
}

template class MTree_wrapper<float>;
template class MTree_wrapper<double>;
template class MTreeSelector_wrapper<float>;
template class MTreeSelector_wrapper<double>;
template class MTreeRangeQuery_wrapper<float>;
template class MTreeRangeQuery_wrapper<double>;
template class MTreeKnnQuery_wrapper<float>;
template class MTreeKnnQuery_wrapper<double>;
