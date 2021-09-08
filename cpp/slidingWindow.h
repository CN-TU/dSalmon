// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

#ifndef DSALMON_SLIDINGWINDOW_H
#define DSALMON_SLIDINGWINDOW_H

#include "MTree.h"
#include "Vector.h"

#include <map>

// Distance based Outlier by Radius
template<typename FloatType=double>
class DBOR {
    // size of the sliding window
    FloatType window;
    // distance threshold for two points to consider as neighbors
    FloatType radius;
    
    // represents one stored data point in the SW
    struct Sample {
        FloatType expire_time;
        int neighbors;
    };

    // we use an M-Tree for performing range queries
    typedef MTree<Vector<FloatType>,Sample,FloatType> Tree;
    Tree tree;
    typedef typename Tree::iterator TreeIterator;
    
  public:
    typedef typename Tree::DistanceFunction DistanceFunction;
    DBOR(FloatType window, FloatType radius, DistanceFunction distance_function=Vector<FloatType>::euclidean, int min_node_size=5, int max_node_size=100, int split_sampling=20) :
        window(window),
        radius(radius),
        tree(distance_function, min_node_size, max_node_size, split_sampling)
    { }
    
    void pruneExpired(FloatType now) {
        // Samples in the tree are stored in order of arrival.
        // Walk the tree and remove expired ones
        while (!tree.empty() && tree.front().second.expire_time <= now) {
            auto& to_prune = tree.front();
            for (auto q = tree.rangeSearch(to_prune.first, radius); !q.atEnd(); ++q)
                (*q).first->second.neighbors--;
            tree.pop_front();
        }
    }

    void fit(const Vector<FloatType>& data, FloatType now) {
        // provide a fit() function for consistency with others,
        // but this can't be made faster than fitPredict()
        fitPredict(data, now);
    }
    
    FloatType fitPredict(const Vector<FloatType>& data, FloatType now) {
        pruneExpired(now);
        int neighbors = 0;
        // count neighbors within radius
        for (auto q = tree.rangeSearch(data, radius); !q.atEnd(); ++q) {
            (*q).first->second.neighbors++;
            neighbors++;
        }
        tree.push_back(std::make_pair(data,Sample{now + window, neighbors}));

        return 1 / (1 + (FloatType)neighbors);
    }
    
    std::size_t windowSize() { return tree.size(); }
    
    // interface for investigating the contents of the SW
    class WindowSample{
        TreeIterator it;
    public:
        WindowSample(TreeIterator it) : it(it) {}
        Vector<FloatType> getSample() { return it->first; }
        FloatType getExpireTime() { return it->second.expire_time; }
        int getNeighbors() { return it->second.neighbors; }
    };
    
    class iterator : public TreeIterator {
      public:
        WindowSample operator*() { return WindowSample(static_cast<TreeIterator>(*this)); };
        iterator() {}
        iterator(TreeIterator it) : TreeIterator(it) {}
    };
    
    iterator begin() { return iterator(tree.begin()); }
    iterator end() { return iterator(tree.end()); }
};


// Distance based Outlier by k nearest neighbors
template<typename FloatType=double>
class SWKNN {
    // size of the sliding window
    FloatType window;
    // number of nearest neighbors to consider
    std::size_t neighbor_cnt;
    // we keep a counter for all processed samples
    int last_index;
    
    // represents one stored data point in the SW
    struct Sample {
        FloatType expire_time;
        // expire_time does not have to increase for each sample. Store
        // an additional index counter as tie breaker to guarantee reproducibility
        int index;
    };

    // we use an M-Tree for performing nearest-neighbor queries
    typedef MTree<Vector<FloatType>,Sample,FloatType> Tree;
    Tree tree;
    typedef typename Tree::iterator TreeIterator;
    
    void pruneExpired(FloatType now) {
        // Samples in the tree are stored in order of arrival.
        // Walk the tree and remove expired samples
        while (!tree.empty() && tree.front().second.expire_time <= now) {
            tree.pop_front();
        }
    }
    
    FloatType fitPredict_impl(const Vector<FloatType>& data, FloatType now, std::vector<FloatType>* neighbors) {
        pruneExpired(now);
        auto breaker = [](const typename Tree::ValueType& a, const typename Tree::ValueType& b) {
            // tie breaker for reproducibility
            return a.second.index < b.second.index;
        };
        auto nearest_neighbors = tree.knnSearch(
            data,
            neighbor_cnt,
            /* sort = */ neighbors != nullptr,
            /* min_radius = */ 0,
            /* max_radius = */ std::numeric_limits<FloatType>::infinity(),
            /* reverse = */ false,
            /* extend_for_ties = */ false,
            breaker
        );
        FloatType score = 0;
        if (neighbors != nullptr) {
            // fill the neighbors vector with information about
            // the distances to nearest neighbors
            neighbors->clear();
            neighbors->reserve(nearest_neighbors.size());
            for (auto& neighbor : nearest_neighbors)
                neighbors->push_back(neighbor.second);
            neighbors->resize(neighbor_cnt, std::numeric_limits<FloatType>::infinity());
            score = (*neighbors)[neighbors->size()-1];
        }
        else if (nearest_neighbors.size() >= neighbor_cnt) {
            // we don't sort nearest_neighbors in this case,
            // so manually find the max distance
            for (auto& neighbor : nearest_neighbors)
                score = std::max(score, neighbor.second);
        }
        else {
            // return infty if we have less points than neighbor_cnt
            score = std::numeric_limits<FloatType>::infinity();
        }
        tree.push_back(std::make_pair(data,Sample{now + window, last_index}));
        last_index++;
        return score;
    }
    
  public:
      typedef typename Tree::DistanceFunction DistanceFunction;
    SWKNN(FloatType window, std::size_t neighbor_cnt, DistanceFunction distance=Vector<FloatType>::euclidean, int min_node_size=5, int max_node_size=100, int split_sampling=20) :
        window(window),
        neighbor_cnt(neighbor_cnt),
        last_index(0),
        tree(distance, min_node_size, max_node_size, split_sampling)
    { }

    void fit(const Vector<FloatType>& data, FloatType now) {
        pruneExpired(now);
        tree.push_back(std::make_pair(data,Sample{now + window, last_index}));
        last_index++;
    }

    FloatType fitPredict(const Vector<FloatType>& data, FloatType now) {
        return fitPredict_impl(data, now, nullptr);
    } 
    
    std::vector<FloatType> fitPredictWithNeighbors(const Vector<FloatType>& data, FloatType now) {
        // perform a fitPredict operation, but additionally return
        // the distances to the neighbor_cnt nearest neighbors
        std::vector<FloatType> scores;
        fitPredict_impl(data, now, &scores);
        return scores;
    }
    
    std::size_t windowSize() { return tree.size(); }
    
    // interface for investigating the SW contents
    class WindowSample{
        TreeIterator it;
    public:
        WindowSample(TreeIterator it) : it(it) {}
        Vector<FloatType> getSample() { return it->first; }
        FloatType getExpireTime() { return it->second.expire_time; }
    };
    
    class iterator : public TreeIterator {
      public:
        WindowSample operator*() { return WindowSample(TreeIterator(*this)); };
        iterator() {}
        iterator(TreeIterator it) : TreeIterator(it) {}
    };
    
    iterator begin() { return iterator(tree.begin()); }
    iterator end() { return iterator(tree.end()); }
};


// Sliding Window Local Outlier Factor
template<typename FloatType=double>
class SWLOF {
    FloatType window;
    std::size_t neighbor_cnt;
    bool simplified;

    int last_index;
    
    struct Sample {
        FloatType expire_time;
        // expire_time does not have to increase for each sample. Store
        // an additional index counter as tie breaker to guarantee reproducibility
        int index;
    };

    typedef MTree<Vector<FloatType>,Sample,FloatType> Tree;
    Tree tree;
    typedef typename Tree::iterator TreeIterator;
    
    void pruneExpired(FloatType now) {
        while (!tree.empty() && tree.front().second.expire_time <= now) {
            tree.pop_front();
        }
    }
    
    // Index caches by points instead of tree indices. This slows down cache lookups a little, 
    // but, due to extend_for_ties=true, it can yield substantial speed-ups if the dataset
    // contains duplicates
    typedef std::map<Vector<FloatType>,std::vector<std::pair<TreeIterator,FloatType>>> NnCache;
    typedef std::map<Vector<FloatType>,std::vector<FloatType>> LrdCache;

    std::vector<std::pair<TreeIterator,FloatType>>& cachedNNQuery(TreeIterator sample, NnCache& cache) {
        int index = sample->second.index;
        auto cache_it = cache.find(sample->first);
        if (cache_it == cache.end()) {
            auto nearest_neighbors = tree.knnSearch(
                sample->first,
                neighbor_cnt + 1,
                /* sort = */ true,
                /* min_radius = */ 0,
                /* max_radius = */ std::numeric_limits<FloatType>::infinity(),
                /* reverse = */ false,
                /* extend_for_ties = */ true
            );
            for (auto it = nearest_neighbors.begin(); it != nearest_neighbors.end(); it++) {
                if (it->first->second.index == index) {
                    nearest_neighbors.erase(it);
                    break;
                }
            }
            std::tie(cache_it,std::ignore) =
                cache.emplace(sample->first, std::move(nearest_neighbors));
        }
        return cache_it->second;
    }
    
    std::vector<FloatType> reachabilityDistance(TreeIterator point_A, TreeIterator point_B, NnCache& nn_cache) {
        FloatType distance = simplified ?
            0 :
            Vector<FloatType>::euclidean(point_A->first, point_B->first);
        std::vector<FloatType> k_distance(1);
        k_distance.reserve(neighbor_cnt);
        FloatType last_distance = -1;
        std::size_t n = 0;
        auto& nearest_neighbors = cachedNNQuery(point_B, nn_cache);
        for (auto& neighbor : nearest_neighbors) {
            if (neighbor.second != last_distance) {
                while (k_distance.size() <= n)
                    k_distance.push_back(k_distance.back());
            }
            k_distance.back() = std::max(distance, neighbor.second);
            last_distance = neighbor.second;
            n++;
        }
        k_distance.resize(neighbor_cnt, k_distance.back());
        return k_distance;
    }
    
    std::vector<FloatType>& localReachabilityDensity(TreeIterator point, LrdCache& lrd_cache, NnCache& nn_cache) {
        auto cache_it = lrd_cache.find(point->first);
        if (cache_it == lrd_cache.end()) {
            std::vector<FloatType> lrd(neighbor_cnt, 0);
            std::vector<int> nearest_neighbor_sizes(neighbor_cnt, 0);
            auto& nearest_neighbors = cachedNNQuery(point, nn_cache);
            // in case of ties, nearest_neighbors.size() might be > neighbor_cnt
            std::size_t n = 0, skip_lrd = 0;
            FloatType last_distance = -1;
            for (auto& neighbor : nearest_neighbors) {
                if (neighbor.second != last_distance) {
                    for (; skip_lrd < n; skip_lrd++)
                        nearest_neighbor_sizes[skip_lrd] = n;
                }
                auto rd = reachabilityDistance(point, neighbor.first, nn_cache);
                for (std::size_t i = skip_lrd; i < neighbor_cnt; i++)
                    lrd[i] += rd[i];
                n++;
                last_distance = neighbor.second;
            }
            for (; skip_lrd < neighbor_cnt; skip_lrd++)
                nearest_neighbor_sizes[skip_lrd] = nearest_neighbors.size();
            for (std::size_t i = 0; i < neighbor_cnt; i++)
                lrd[i] = nearest_neighbor_sizes[i] / lrd[i];
            std::tie(cache_it, std::ignore) = 
                lrd_cache.emplace(point->first, std::move(lrd));
        }
        return cache_it->second;
    }
    
  public:
      typedef typename Tree::DistanceFunction DistanceFunction;
    SWLOF(FloatType window, std::size_t neighbor_cnt, bool simplified, DistanceFunction distance=Vector<FloatType>::euclidean, int min_node_size=5, int max_node_size=100, int split_sampling=20) :
        window(window),
        neighbor_cnt(neighbor_cnt),
        simplified(simplified),
        last_index(0),
        tree(distance, min_node_size, max_node_size, split_sampling)
    { }
    
    void fit(const Vector<FloatType>& data, FloatType now) {
        pruneExpired(now);
        tree.push_back(std::make_pair(data,Sample{now + window, last_index}));
        last_index++;
    }
    
    std::vector<FloatType> fitPredict(const Vector<FloatType>& data, FloatType now) {
        pruneExpired(now);
        LrdCache lrd_cache;
        NnCache nn_cache;

        tree.push_back(std::make_pair(data,Sample{now + window, last_index}));
        auto tree_it = std::prev(tree.end());
        
        if (tree.size() == 1) {
            // if we have no other points, arbitrarily define this an inlier.
            last_index++;
            return std::vector<FloatType>(neighbor_cnt, 0);
        }
            
        auto& nearest_neighbors = cachedNNQuery(tree_it, nn_cache);
        std::vector<FloatType> lof(neighbor_cnt, 0);
        std::vector<std::size_t> nearest_neighbor_sizes(neighbor_cnt, 0);
        std::size_t n = 0, skip_lof = 0;
        FloatType last_distance = -1;
        // in case of ties, nearest_neighbors.size() might be > neighbor_cnt
        for (auto& neighbor : nearest_neighbors) {
            if (neighbor.second != last_distance) {
                for (; skip_lof < n; skip_lof++)
                    nearest_neighbor_sizes[skip_lof] = n;
            }
            auto& lrd = localReachabilityDensity(neighbor.first, lrd_cache, nn_cache);
            for (std::size_t i = skip_lof; i < neighbor_cnt; i++)
                lof[i] += lrd[i];
            n++;
            last_distance = neighbor.second;
        }
        for (; skip_lof < neighbor_cnt; skip_lof++)
            nearest_neighbor_sizes[skip_lof] = nearest_neighbors.size();
        auto& lrd = localReachabilityDensity(tree_it, lrd_cache, nn_cache);
        for (std::size_t i = 0; i < neighbor_cnt; i++)
            lof[i] /= (nearest_neighbor_sizes[i] * lrd[i]);
        
        last_index++;
        return lof;
    }
    
    std::size_t windowSize() { return tree.size(); }
    
    // interface for investigating the SW contents
    class WindowSample{
        TreeIterator it;
    public:
        WindowSample(TreeIterator it) : it(it) {}
        Vector<FloatType> getSample() { return it->first; }
        FloatType getExpireTime() { return it->second.expire_time; }
    };
    
    class iterator : public TreeIterator {
      public:
        WindowSample operator*() { return WindowSample(TreeIterator(*this)); };
        iterator() {}
        iterator(TreeIterator it) : TreeIterator(it) {}
    };
    
    iterator begin() { return iterator(tree.begin()); }
    iterator end() { return iterator(tree.end()); }
};


#endif
