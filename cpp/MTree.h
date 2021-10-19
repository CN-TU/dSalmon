// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

#ifndef DSALMON_MTREE_H
#define DSALMON_MTREE_H

#include <functional>
#include <limits>
#include <list>
#include <stack>
#include <random>
#include <utility>
#include <vector>
#include <assert.h>

#include <map>
#include <mutex>

#include "util.h"

// Node statistics to simply count all descendants
template<typename Key, typename T>
class MTreeDescendantCounter {
    typedef std::pair<const Key,T> ValueType;
    std::size_t descendants;
  public:
    MTreeDescendantCounter() : descendants(0) {}
    std::size_t getDescendantCount() const {
        return descendants;
    }
    void addDescendant(const ValueType& value) {
        descendants++;
    }
    void addDescendants(const MTreeDescendantCounter& from) {
        descendants += from.descendants;
    }
    void removeDescendants(const MTreeDescendantCounter& from) {
        descendants -= from.descendants;
    }
    void removeDescendant(const ValueType& value) {
        descendants--;
    }
    void serialize(std::ostream& os) {
        serializeInt<uint64_t>(os, descendants);
    }
    void unserialize(std::istream& is) {
        descendants = unserializeInt<uint64_t>(is);
    }
};


// An M-Tree is a tree-based data structure which allows efficient range
// and nearest-neighbor search assuming a distance function satisfying
// the triangle inequality.
template<typename Key, typename T, typename DistanceType = double, typename NodeStats = MTreeDescendantCounter<Key,T>>
class MTree {
  public:
    typedef Key KeyType;
    typedef T MappedType;
    typedef std::pair<const Key, T> ValueType;
    typedef NodeStats NodeStatsType;
    
    typedef std::function<DistanceType(const Key&, const Key&)> DistanceFunction;
    typedef void *NodeTagType;
    
    class Query;
    class RangeQuery;
    
    // A BoundEstimator receives lower and upper bounds for distances of all descendants of a particular node
    // and uses them to narrow down the search radius. Together with custom NodeStats, this allows
    // highly efficient filtering knn queries.
    typedef std::function<void(Query&,const NodeStats*,NodeTagType,NodeTagType,DistanceType,DistanceType)> BoundEstimator;
    
  private:
    // min. number of children of RoutingNodes
    std::size_t min_node_size;
    // max. number of children of RoutingNodes
    std::size_t max_node_size;
    // number of key combinations to try when splitting a node
    std::size_t split_sampling;
    std::mt19937 rng;
    
    // Neighbor queries for an M-Tree work by pruning nodes which are
    // too far away. To avoid numerical issues, make the upper bound
    // for pruning slightly larger and the lower bound slightly smaller
    // than theoretically possible.
    constexpr static DistanceType lower_bound_factor = 0.999;
    constexpr static DistanceType upper_bound_factor = 1.001;
    
    DistanceFunction distance_function;
    
    struct RoutingNode;
    
    struct Node {
        // parent node in the tree
        RoutingNode* parent;
        // distance from the parent's center
        DistanceType parent_distance;
        // center of this node
        virtual const KeyType& getKey() = 0;
        Node(RoutingNode *parent, DistanceType parent_distance) :
            parent(parent),
            parent_distance(parent_distance)
        { }
    };
    
    // ObjectNodes represent entries of the tree
    struct ObjectNode final : public Node {
        ValueType value;
        const KeyType& getKey() override { return value.first; }
        // store list iterator for being able to efficiently delete nodes
        typename std::list<ObjectNode>::iterator list_ref;

        ObjectNode(ValueType&& value) :
            Node(nullptr,0),
            value(std::move(value))
        { }
    };
    
    // RoutingNodes are used for subsuming several nodes
    // within one hyper sphere
    struct RoutingNode final : public Node {
        std::vector<Node*> children;
        // mutex used for parallel inserting
        std::mutex mutex;
        // is_leaf==true <=> children are ObjectNode's
        bool is_leaf;
        // center of this node
        KeyType key;
        const KeyType& getKey() override { return key; }
        // Statistics. Counts descendants by default
        NodeStats stats;
        // descendant which is furthest away from
        // key and its distance to key
        DistanceType covering_radius;
        ObjectNode *furthest_descendant;
        
        // TODO: streamline constructor calls?
        RoutingNode() : Node{nullptr, 0} {}
        RoutingNode(RoutingNode* parent, bool is_leaf) : 
            Node{parent, 0},
            is_leaf(is_leaf)
        { }
    };

    typedef std::list<ObjectNode> EntryList;
    typedef typename EntryList::iterator EntryListIterator;

    EntryList entries;     // protected by root_mutex
    RoutingNode *root;

    std::mutex root_mutex;

    RoutingNode& addRoutingNode(RoutingNode& parent, bool is_leaf);
    bool isDescendant(Node& child, RoutingNode& parent);
    void computeNodeRadius(RoutingNode& node);
    void computeNonLeafRadius(RoutingNode& node, RoutingNode& from, bool only_increase);

    void insertIterator(EntryListIterator it, std::mutex** last_mutex);
    void treeInsert(ObjectNode& new_entry, std::mutex** last_mutex);
    void split(RoutingNode& node);
    void promoteAndPartition(std::vector<Node*>& children, RoutingNode& node1, RoutingNode& node2);
    std::pair<const KeyType&,const KeyType&> promote(const std::vector<Node*>& children);
    DistanceType partition(bool from_leaf, const std::vector<Node*> from, RoutingNode& to_1, RoutingNode& to_2, DistanceType estimated_radius_bound);

    void treeErase(ObjectNode& to_erase);
    void pullUpRoot();
    void rebalanceNode(RoutingNode& node);
    void donateChild(RoutingNode& from, RoutingNode& to);
    void mergeRoutingNodes(RoutingNode& from, RoutingNode& to, DistanceType from_to_distance);
    
    Query subtreeSearch(const KeyType& needle, DistanceType min_radius, DistanceType max_radius, bool reverse, BoundEstimator estimator, RoutingNode& subtree, bool locking) {
        return Query(this, needle, min_radius, max_radius, reverse, estimator, &subtree, locking);
    }
    
    template<typename BaseIterator>
    class IteratorTempl : public BaseIterator {
      public:
        ValueType& operator*() const { return BaseIterator::operator*().value; }
        ValueType* operator->() const { return &BaseIterator::operator*().value; }
        IteratorTempl() {}
        IteratorTempl(BaseIterator it) : BaseIterator(it) {}
    };
    
    // TODO: provide default bound estimator instead
    // BoundEstimator which does nothing
    struct NopBoundEstimator {
        void operator()(const NodeStats*, DistanceType, DistanceType) {}
    };
    // A TieBreaker allows the order of knn queries to be specified if points
    // have the same distance to the search query.
    // TODO: additional provide distance?
    // TieBreaker which does nothing (random results for ties)
    struct NopTieBreaker {
        bool operator() (const ValueType&, const ValueType&) { return false; }
    };
    void check();
    
  public:
    typedef IteratorTempl<typename EntryList::iterator> iterator;
    typedef IteratorTempl<typename EntryList::const_iterator> const_iterator;
    
  private:
    std::pair<iterator,DistanceType> nnSubtreeSearch(const KeyType& needle, DistanceType min_radius, DistanceType max_radius, bool reverse, RoutingNode& subtree, bool locking);
    template<typename TieBreaker>
    std::vector<std::pair<iterator,DistanceType>> knnSubtreeSearch(const KeyType& needle, unsigned k, bool sort, DistanceType min_radius, DistanceType max_radius, bool reverse, TieBreaker tie_breaker, bool extend_for_ties, RoutingNode& subtree);

  public:
    MTree(DistanceFunction distance_function, std::size_t min_node_size=5, std::size_t max_node_size=100, std::size_t split_sampling=20) :
        min_node_size(min_node_size),
        max_node_size(max_node_size),
        split_sampling(split_sampling),
        distance_function(distance_function),
        root(nullptr)
    { }
        
    MTree(const MTree& other) : root(nullptr) { *this = other; }
    MTree& operator=(const MTree&);
    MTree(MTree&&) = default;
    MTree& operator=(MTree&&) = default;
    
    ~MTree() { clear(); }

    iterator begin() { return iterator(entries.begin()); }
    const_iterator cbegin() const { return const_iterator(entries.cbegin()); }
    iterator end() { return iterator(entries.end()); }
    const_iterator cend() const { return const_iterator(entries.cend()); }
    
    ValueType& front() { return entries.front().value; }
    const ValueType& front() const { return entries.front().value; }
    ValueType& back() { return entries.back().value; }
    const ValueType& back() const { return entries.back().value; }
    
    std::size_t size() const { return entries.size(); }
    bool empty() const { return entries.empty(); }

    void clear();
    
    iterator insert(iterator pos, ValueType&& value);
    void push_back(ValueType&& value) { insert(end(), std::move(value)); }
    void push_front(ValueType&& value) { insert(begin(), std::move(value)); }
    void erase(iterator pos); // TODO: change to const_iterator
    void pop_back() { erase(std::prev(end())); }
    void pop_front() { erase(begin()); }
    
    void reverse() { entries.reverse(); }
    
    template<typename Modifier>
    void modify(iterator pos, Modifier modifier);
    // only modify stats while leaving key unmodified
    // void modifyStats(iterator pos, Modifier modifier);    
    
    Query search(const KeyType& needle, DistanceType min_radius = 0, DistanceType max_radius = std::numeric_limits<DistanceType>::infinity(), bool reverse = false, BoundEstimator estimator = NopBoundEstimator()) {
        return subtreeSearch(needle, min_radius, max_radius, reverse, estimator, *root, false);
    }
    RangeQuery rangeSearch(const KeyType& needle, DistanceType radius) {
        return RangeQuery(this, needle, radius);
    }
    template<typename TieBreaker = NopTieBreaker>
    std::vector<std::pair<iterator,DistanceType>> knnSearch(const KeyType& needle, unsigned k, bool sort = false, DistanceType min_radius = 0, DistanceType max_radius = std::numeric_limits<DistanceType>::infinity(), bool reverse = false, bool extend_for_ties = false, TieBreaker tie_breaker = NopTieBreaker()) {
        if (root == nullptr)
            return {};
        return knnSubtreeSearch(needle, k, sort, min_radius, max_radius, reverse, tie_breaker, extend_for_ties, *root);
    }

    template<typename KeySer, typename TSer>
    void serialize(std::ostream& out, KeySer& key_ser = KeySer{}, TSer& T_ser = TSer{});
    template<typename KeyUnser, typename TUnser>
    void unserialize(std::istream& in, KeyUnser& key_unser = KeyUnser{}, TUnser& T_unser = TUnser{});
};


template<typename Key, typename T, typename DistanceType, typename NodeStats>
class MTree<Key,T,DistanceType,NodeStats>::RangeQuery {
    MTree* tree;
    KeyType needle;
    DistanceType radius;
    
    RoutingNode* current_leaf;
    DistanceType current_leaf_distance;
    typename std::vector<Node*>::iterator leaf_it;
    iterator last_returned;
    DistanceType last_distance;
    bool is_at_end;

    std::stack<std::pair<RoutingNode*,DistanceType>> queue;
    
    void findNextLeaf();
    iterator advance();
    
  public:
    RangeQuery& operator++();
    std::pair<iterator,DistanceType> operator*() {
        return std::make_pair(last_returned, last_distance);
    }
    bool atEnd() { return is_at_end; }
    
    RangeQuery() : is_at_end(true) {}
    
// private to MTree: // TODO
    RangeQuery(MTree* tree, const KeyType& needle, DistanceType radius) :
        tree(tree),
        needle(needle),
        radius(radius),
        current_leaf(nullptr),
        is_at_end(false)
    {
        if (tree->root == nullptr) {
            is_at_end = true;
        }
        else {
            queue.emplace(
                tree->root,
                tree->distance_function(tree->root->getKey(), needle)
            );
            ++(*this);
        }
    }
};


template<typename Key, typename T,typename DistanceType, typename NodeStats>
class MTree<Key,T,DistanceType,NodeStats>::Query {
    MTree* tree;
    KeyType needle; // TODO: make this a reference?
    DistanceType min_search_radius;
    DistanceType tolerant_min_search_radius;
    DistanceType max_search_radius;
    DistanceType tolerant_max_search_radius;

    BoundEstimator bound_estimator;
    bool reverse;
    RoutingNode* subtree;
    bool locking;
    
    RoutingNode* current_leaf;
    DistanceType current_leaf_distance;
    typename std::vector<Node*>::iterator leaf_it;
    iterator last_returned;
    DistanceType last_distance;
    
    struct QueueEntry {
        RoutingNode* node;
        DistanceType center_distance;
        DistanceType distance_bound;
        QueueEntry(RoutingNode* node, DistanceType center_distance, DistanceType distance_bound) :
            node(node),
            center_distance(center_distance),
            distance_bound(distance_bound)
        { }
    };
    struct QueueEntryCmp {
        bool reverse;
        bool operator() (const QueueEntry& a, const QueueEntry& b) {
            return reverse ? a.distance_bound < b.distance_bound : a.distance_bound > b.distance_bound;
        }
    } queue_cmp;
    
    std::vector<QueueEntry> queue;
    bool pruneQueue(RoutingNode** node, DistanceType* distance);
    void findNextLeaf();
    bool is_at_end;
    
  public:
    // min radius must not decrease, otherwise results are undefined.
    void setMinRadius(DistanceType radius) {
        assert(radius >= min_search_radius);
        min_search_radius = radius;
        tolerant_min_search_radius = lower_bound_factor * radius;
    }
    // max radius must not increase, otherwise results are undefined.
    void setMaxRadius(DistanceType radius) {
        assert (radius <= max_search_radius);
        max_search_radius = radius;
        tolerant_max_search_radius = upper_bound_factor * radius;
    }
    
    DistanceType getMinRadius() { return min_search_radius; }
    DistanceType getMaxRadius() { return max_search_radius; }
    
    Query& operator++();
    std::pair<iterator,DistanceType> operator*() {
        return std::make_pair(last_returned, last_distance);
    }
    bool atEnd() { return is_at_end; }
    
    Query() : is_at_end(true) {}
    
// private to MTree:
    Query(MTree* mtree, const KeyType& needle, DistanceType min_radius, DistanceType max_radius, bool reverse, BoundEstimator estimator, RoutingNode* subtree, bool locking) :
        tree(mtree),
        needle(needle),
        min_search_radius(min_radius),
        tolerant_min_search_radius(lower_bound_factor * min_radius),
        max_search_radius(max_radius),
        tolerant_max_search_radius(upper_bound_factor * max_radius),
        bound_estimator(estimator),
        reverse(reverse),
        subtree(subtree),
        locking(locking),
        current_leaf(nullptr),
        is_at_end(false)
    {
        queue_cmp.reverse = reverse;
        if (subtree == nullptr) {
            is_at_end = true;
        }
        else {
            queue.emplace_back(
                subtree,
                tree->distance_function(subtree->getKey(), needle),
                reverse ? std::numeric_limits<DistanceType>::infinity() : 0
            );
            ++(*this);
        }
    }
};

#include "MTree_impl.h"

#endif

