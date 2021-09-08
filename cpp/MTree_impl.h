// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

#include <algorithm>
#include <assert.h>
#include <unordered_map>
#include <unordered_set>

#include <thread>

#include "PlaceholderQueue.h"

template<typename Key, typename T, typename DistanceType, typename NodeStats>
void MTree<Key,T,DistanceType,NodeStats>::clear() {
    // Walk the tree and delete all RoutingNode's. Memory of
    // ObjectNode's is managed by the entries list
    RoutingNode* node = root;
    while (node != nullptr) {
        if (node->is_leaf || node->children.empty()) {
            RoutingNode* parent = node->parent;
            delete node;
            node = parent;
        }
        else {
            Node* child = node->children.back();
            node->children.pop_back();
            node = static_cast<RoutingNode*>(child);
        }
    }
    entries.clear();
    root = nullptr;
}

template<typename Key, typename T, typename DistanceType, typename NodeStats>
auto MTree<Key,T,DistanceType,NodeStats>::operator=(const MTree& other) -> MTree& {
    clear();
    min_node_size = other.min_node_size;
    max_node_size = other.max_node_size;
    split_sampling = other.split_sampling;
    rng = other.rng;
    distance_function = other.distance_function;
    entries = other.entries;
    
    std::unordered_map<Node*,Node*> pointers;
    
    auto entry_to = entries.begin();
    for (ObjectNode& entry_from : other.entries) {
        pointers[&entry_from] = &*entry_to;
        entry_to->list_ref = entry_to;
        ++entry_to;
        RoutingNode* parent = entry_from.parent;
        while (parent != nullptr && !pointers.count(parent)) {
            pointers[parent] = new RoutingNode(*parent);
            parent = parent->parent;
        }
    }
    // now walk all nodes and update pointers
    for (auto& mapping : pointers) {
        Node* node = mapping.second;
        node->parent = static_cast<RoutingNode*>(pointers[node->parent]);
        RoutingNode* node_as_RoutingNode = dynamic_cast<RoutingNode*>(node);
        if (node_as_RoutingNode != nullptr) {
            node_as_RoutingNode->furthest_descendant =
                static_cast<ObjectNode*>(pointers[node_as_RoutingNode->furthest_descendant]);
            for (Node*& child : node_as_RoutingNode->children)
                child = pointers[child];
        }
    }
    root = static_cast<RoutingNode*>(pointers[other.root]);
    return *this;
}

template<typename Key, typename T, typename DistanceType, typename NodeStats>
bool MTree<Key,T,DistanceType,NodeStats>::isDescendant(Node& child, RoutingNode& parent) {
    assert(!child.parent || std::find(child.parent->children.begin(), child.parent->children.end(), &child) != child.parent->children.end());
    for (RoutingNode *node = child.parent; node != nullptr; node = node->parent) {
        if (node == &parent) {
            return true;
        }
        assert(!node->parent || std::find(node->parent->children.begin(), node->parent->children.end(), node) != node->parent->children.end());
    }
    return false;
}

template<typename Key, typename T, typename DistanceType, typename NodeStats>
void MTree<Key,T,DistanceType,NodeStats>::computeNonLeafRadius(RoutingNode& node, RoutingNode& from, bool only_increase) {
    if (&node != root) {
        std::pair<iterator,DistanceType> furthest = nnSubtreeSearch(
            node.getKey(), // needle
            only_increase ? node.covering_radius : 0, // min search radius
            std::numeric_limits<DistanceType>::infinity(), // max search radius
            true, // find outermost point instead of nearest
            from,
            true
        );
        if (furthest.first != end()) {
            auto entry_it =
                static_cast<EntryListIterator>(furthest.first);
            node.furthest_descendant = &*entry_it;
            node.covering_radius = furthest.second;
        }
        else {
            assert (only_increase);
        }
    }
}

template<typename Key, typename T, typename DistanceType, typename NodeStats>
void MTree<Key,T,DistanceType,NodeStats>::computeNodeRadius(RoutingNode& node) {
    // no need to keep track of covering radius of root
    if (&node != root) {
        // A node is allowed to have slightly less than min_node_size children at this point
        assert (node.children.size() >= min_node_size - 1);
        if (node.is_leaf) {
            DistanceType max_distance = std::numeric_limits<DistanceType>::lowest();
            ObjectNode* new_furthest = nullptr;
            for (Node* child : node.children) {
                if (child->parent_distance > max_distance) {
                    max_distance = child->parent_distance;
                    new_furthest = static_cast<ObjectNode*>(child);
                }
            }
            assert(new_furthest != nullptr);
            node.furthest_descendant = new_furthest;
            node.covering_radius = max_distance;
        }
        else {
            computeNonLeafRadius(node, node, false);
        }
    }
}

template<typename Key, typename T, typename DistanceType, typename NodeStats>
auto MTree<Key,T,DistanceType,NodeStats>::addRoutingNode(RoutingNode& parent, bool is_leaf) -> RoutingNode& {
    RoutingNode* new_routing = new RoutingNode(&parent, is_leaf);
    parent.children.push_back(new_routing);
    return *new_routing;
}

template<typename Key, typename T, typename DistanceType, typename NodeStats>
void MTree<Key,T,DistanceType,NodeStats>::check() {
#if !defined(NDEBUG)
    static std::mt19937 check_rng;
    if (check_rng() - check_rng.min() > 0.000001 * (check_rng.max() - check_rng.min()))
        return;
    if (root == nullptr || entries.empty()) {
        assert(root == nullptr);
        assert(entries.empty());
        return;
    }
    std::unordered_set<const Node*> pointers;
    std::unordered_set<const RoutingNode*> routing_nodes;
    for (auto& entry : entries) {
        RoutingNode* parent = entry.parent;
        assert(entry.parent_distance == distance_function(entry.parent->getKey(), entry.getKey()));
        for(; parent != root; parent = parent->parent) {
            assert(parent->covering_radius >= distance_function(parent->getKey(), entry.getKey()));
        }
        while (parent != nullptr && !routing_nodes.count(parent)) {
            routing_nodes.insert(parent);
            parent = parent->parent;
        }
        pointers.insert(&entry);
    }
    for (RoutingNode* node : routing_nodes) {
        if (node != root)  {
            assert(node->parent_distance == distance_function(node->parent->getKey(), node->getKey()));
        }
        pointers.insert(node);
    }
    std::stack<RoutingNode*> queue;

    assert(pointers.count(root)==1);
    queue.push(root);
    std::size_t total_children = 0;
    std::size_t total_routingnodes = 1;
    while (!queue.empty()) {
        RoutingNode* node = queue.top();
        queue.pop();
        assert (node == root || pointers.count(node->furthest_descendant) == 1);
        for (Node* child : node->children) {
            assert (child->parent == node);
            assert (pointers.count(child) == 1);
        }
        if (node != root) {
            std::pair<iterator,DistanceType> furthest = nnSubtreeSearch(
                node->getKey(), // needle
                0, // min search radius
                std::numeric_limits<DistanceType>::infinity(), // max search radius
                true, // find outermost point instead of nearest
                *node,
                false
            );
            DistanceType furthest_dist = distance_function(node->furthest_descendant->getKey(), node->getKey());
            assert(isDescendant(*node->furthest_descendant, *node));
            assert(furthest_dist == node->covering_radius);
            assert(node->covering_radius == furthest.second);
        }
        if (node->is_leaf) {
            total_children += node->children.size();
            assert(node->stats.getDescendantCount() == node->children.size());
        }
        else {
            total_routingnodes += node->children.size();
            std::size_t descendants = 0;
            for (Node* child : node->children) {
                descendants += static_cast<RoutingNode*>(child)->stats.getDescendantCount();
                queue.push(static_cast<RoutingNode*>(child));
            }
            assert(node->stats.getDescendantCount() == descendants);
        }
    }
    assert (total_routingnodes == routing_nodes.size());
    assert(total_children == entries.size());
    assert(root->stats.getDescendantCount() == total_children);
#endif
}

// TODO: make sure a node center does not move away from points too much
template<typename Key, typename T, typename DistanceType, typename NodeStats>
void MTree<Key,T,DistanceType,NodeStats>::promoteAndPartition(std::vector<Node*>& children, RoutingNode& node1, RoutingNode& node2) {
    assert(node1.is_leaf == node2.is_leaf);

    KeyType best_key_1, best_key_2;
    DistanceType best_av_radius = std::numeric_limits<DistanceType>::infinity();

    // TODO: maybe make sure we never try the same combination twice
    for (std::size_t i = 0; i < split_sampling; i++) {
        KeyType Op1, Op2;
        std::tie(node1.key, node2.key) = promote(children);
        DistanceType estimated_av_radius =
            partition(node1.is_leaf, children, node1, node2, best_av_radius);
        if (estimated_av_radius < best_av_radius) {
            std::swap(node1.key, best_key_1);
            std::swap(node2.key, best_key_2);
            best_av_radius = estimated_av_radius;
        }
    }
    std::swap(node1.key, best_key_1);
    std::swap(node2.key, best_key_2);

    if (node1.is_leaf) {
        for (Node* child : node1.children)
            node1.stats.addDescendant(static_cast<ObjectNode*>(child)->value);
        for (Node* child : node2.children)
            node2.stats.addDescendant(static_cast<ObjectNode*>(child)->value);
    }
    else {
        for (Node* child : node1.children)
            node1.stats.addDescendants(static_cast<RoutingNode*>(child)->stats);
        for (Node* child : node2.children)
            node2.stats.addDescendants(static_cast<RoutingNode*>(child)->stats);
    }
    computeNodeRadius(node1);
    computeNodeRadius(node2);
}

template<typename Key, typename T, typename DistanceType, typename NodeStats>
void MTree<Key,T,DistanceType,NodeStats>::split(RoutingNode& to_split) {
    for (RoutingNode *node = &to_split; ; node = node->parent) {
        std::size_t children_cnt = node->children.size();
        (void)children_cnt; // only used for asserts. suppress warning
        std::vector<Node*> old_list;
        std::swap(old_list, node->children);
        if (node != root) {
            assert (node->parent != nullptr);
            RoutingNode& new_routing =
                addRoutingNode(*node->parent, node->is_leaf);
            node->stats = {};
            promoteAndPartition(old_list, *node, new_routing);
            node->parent_distance = distance_function(node->getKey(), node->parent->getKey());
            new_routing.parent_distance = distance_function(new_routing.getKey(), node->parent->getKey());
            assert(children_cnt == (node->children.size() + new_routing.children.size()));
            if (node->parent->children.size() <= max_node_size)
                return;
        }
        else {
            assert(node->parent == nullptr);
            RoutingNode& new_1 = addRoutingNode(*root, root->is_leaf);
            RoutingNode& new_2 = addRoutingNode(*root, root->is_leaf);
            promoteAndPartition(old_list, new_1, new_2);
            new_1.parent_distance = distance_function(new_1.getKey(), root->getKey());
            new_2.parent_distance = distance_function(new_2.getKey(), root->getKey());
            // TODO: maybe update root key from time to time?
            assert(children_cnt == (new_1.children.size() + new_2.children.size()));
            root->is_leaf = false;
            return;
        }
    }
}

template<typename Key, typename T, typename DistanceType, typename NodeStats>
auto MTree<Key,T,DistanceType,NodeStats>::partition(bool from_leaf, const std::vector<Node*> from, RoutingNode& to_1, RoutingNode& to_2, DistanceType estimated_radius_bound) -> DistanceType {
    struct Child {
        std::size_t index;
        DistanceType from_1;
        DistanceType from_2;
        DistanceType quotient;
    };
    std::vector<Child> distances;
    distances.reserve(from.size());
    for (std::size_t i = 0; i < from.size(); i++) {
        DistanceType d1 = distance_function(from[i]->getKey(), to_1.getKey());
        DistanceType d2 = distance_function(from[i]->getKey(), to_2.getKey());
        distances.push_back({i,d1,d2,d1/d2});
    }
    std::sort(distances.begin(), distances.end(), [](const Child& a, const Child& b){
        return a.quotient < b.quotient;
    });
    std::size_t boundary = from.size()/2;
    if (distances[boundary].quotient > 1) {
        while (distances[boundary].quotient > 1 && boundary > min_node_size)
            boundary--;
    }
    else {
        while (distances[boundary].quotient < 1 && boundary < from.size() - min_node_size)
            boundary++;
    }
    DistanceType estimated_radius_1 = 0, estimated_radius_2 = 0;
    for (std::size_t i = 0; i < from.size(); i++) {
        Node* node = from[distances[i].index];
        DistanceType parent_distance = (i < boundary) ? distances[i].from_1 : distances[i].from_2;
        DistanceType estimated_radius = from_leaf ?
            parent_distance :
            (parent_distance + static_cast<RoutingNode*>(node)->covering_radius) ;
        if (i < boundary)
            estimated_radius_1 = std::max(estimated_radius_1, estimated_radius);
        else
            estimated_radius_2 = std::max(estimated_radius_2, estimated_radius);
    }
    if (estimated_radius_1 + estimated_radius_2 >= estimated_radius_bound)
        return estimated_radius_1 + estimated_radius_2;
    to_1.children.clear();
    to_2.children.clear();
    for (std::size_t i = 0; i < from.size(); i++) {
        Node* node = from[distances[i].index];
        RoutingNode& to = (i < boundary) ? to_1 : to_2;
        node->parent = &to;
        node->parent_distance = (i < boundary) ? distances[i].from_1 : distances[i].from_2;
        to.children.push_back(node);
    }        
    return estimated_radius_1 + estimated_radius_2;
}

// template<typename Key, typename T, typename DistanceType, typename NodeStats>
// void MTree<Key,T,DistanceType,NodeStats>::partition(std::vector<Node*> from, RoutingNode& to_1, RoutingNode& to_2) {
    // for (Node* node : from) {
        // DistanceType d1 = distance_function(node->getKey(), to_1.getKey());
        // DistanceType d2 = distance_function(node->getKey(), to_2.getKey());
        // RoutingNode& to = (d1 < d2) ? to_1 : to_2;
        // node->parent_distance = std::min(d1,d2);
        // to.children.push_back(node);
        // if (from_leaf)
            // to.stats.addDescendant(static_cast<ObjectNode*>(node)->value);
        // else
            // to.stats.addDescendants(static_cast<RoutingNode*>(node)->stats);
    // }
    // recomputeParentRadius(to_1);
    // recomputeParentRadius(to_2);
// }
    
template<typename Key, typename T, typename DistanceType, typename NodeStats>
std::pair<const Key&,const Key&> MTree<Key,T,DistanceType,NodeStats>::promote(const std::vector<Node*>& children) {
    // std::uniform_int_distribution gives better randomness, but is also slower.
    // Here we don't need perfect randomness.
    std::size_t a = rng() % children.size();
    std::size_t b = rng() % (children.size()-1);
    if (b >= a) {
        b++;
    }
    else {
        std::swap(a,b);
    }
    return std::pair<const KeyType&, const KeyType&>(
        children[a]->getKey(),
        children[b]->getKey()
    );
}

template<typename Key, typename T, typename DistanceType, typename NodeStats>
void MTree<Key,T,DistanceType,NodeStats>::treeInsert(ObjectNode& new_entry, std::mutex** last_mutex) {
    const KeyType& key = new_entry.value.first;
    RoutingNode *node = root;
    DistanceType nearest_distance = 0;
    assert(root != nullptr);
    root->stats.addDescendant(new_entry.value);
    while (!node->is_leaf) {
        bool within_covering_radius = false;
        Node *nearest = nullptr;
        DistanceType radius_increase = std::numeric_limits<DistanceType>::infinity();
        nearest_distance = std::numeric_limits<DistanceType>::infinity();

        for (Node* child : node->children) {
            RoutingNode& child_as_RoutingNode = static_cast<RoutingNode&>(*child);
            DistanceType parent_distance = distance_function(child_as_RoutingNode.key, key);
            
            if (!within_covering_radius && parent_distance <= child_as_RoutingNode.covering_radius) {
                within_covering_radius = true;
                nearest = child;
                nearest_distance = parent_distance;
            }
            else if (within_covering_radius) {
                if (parent_distance <= child_as_RoutingNode.covering_radius) {
                    if (parent_distance < nearest_distance) {
                        nearest = child;
                        nearest_distance = parent_distance;
                    }
                }
            }
            else {
                assert(parent_distance > child_as_RoutingNode.covering_radius);
                if (parent_distance < child_as_RoutingNode.covering_radius + radius_increase) {
                    nearest = child;
                    nearest_distance = parent_distance;
                    radius_increase = parent_distance - child_as_RoutingNode.covering_radius;
                }
            }
        }
        assert(nearest != nullptr);
        node = static_cast<RoutingNode*>(nearest);
        if (!within_covering_radius) {
            node->covering_radius = nearest_distance;
            node->furthest_descendant = &new_entry;
        }
        node->mutex.lock();
        node->stats.addDescendant(new_entry.value);
        if (node->children.size() < max_node_size) {
            (*last_mutex)->unlock();
            *last_mutex = &node->mutex;
        }
        else {
            // As we keep a mutex of a higher node locked, node->mutex
            // can immediately be freed again. However, locking
            // node->mutex cannot be omitted, as we have to ensure
            // that previous inserts are done.
            node->mutex.unlock();
        }
    }
    assert( (node == root) == root->is_leaf );
    new_entry.parent = node;
    new_entry.parent_distance =
        (node == root) ? distance_function(root->getKey(), key) : nearest_distance;
    node->children.push_back(&new_entry);
}

template<typename Key, typename T, typename DistanceType, typename NodeStats>
void MTree<Key,T,DistanceType,NodeStats>::insertIterator(EntryListIterator it, std::mutex** last_mutex) {
    if (root != nullptr) {
        assert(entries.size() > 1);
        treeInsert(*it, last_mutex);
        if (it->parent->children.size() > max_node_size)
            split(*it->parent);
        assert(it->parent->children.size() <= max_node_size);
    }
    else {
        assert (entries.size() == 1);
        root = new RoutingNode(nullptr, true);
        root->key = it->value.first;
        root->stats.addDescendant(it->value);
        it->parent = root;
        it->parent_distance = 0;
        // no need to keep track of covering radius of root
        root->covering_radius = 0;
        root->furthest_descendant = nullptr;
        root->children.push_back(&*it);
    }
    check();
}

template<typename Key, typename T, typename DistanceType, typename NodeStats>
auto MTree<Key,T,DistanceType,NodeStats>::insert(iterator pos, ValueType&& value) -> iterator {
    EntryListIterator it;
    std::mutex* last_mutex = &root_mutex;
    // while (!root_mutex.try_lock()) std::this_thread::yield();

    root_mutex.lock();
    it = entries.emplace(pos, std::move(value));
    it->list_ref = it;
    insertIterator(it, &last_mutex);
    last_mutex->unlock();
    return iterator{it};
}

template<typename Key, typename T, typename DistanceType, typename NodeStats>
template<typename Modifier>
void MTree<Key,T,DistanceType,NodeStats>::modify(iterator pos, Modifier modifier) {
    EntryListIterator it = static_cast<EntryListIterator>(pos);
    treeErase(*it);
    // TODO: get rid of const_cast
    modifier(const_cast<Key&>(pos->first), pos->second);
    std::mutex* last_mutex = &root_mutex;
    insertIterator(it, &last_mutex);
}

template<typename Key, typename T, typename DistanceType, typename NodeStats>
void MTree<Key,T,DistanceType,NodeStats>::donateChild(RoutingNode& from, RoutingNode& to) {
    assert (from.parent == to.parent);
    assert (from.is_leaf == to.is_leaf);
    std::vector<Node*>& grandchildren = from.children;
    typename std::vector<Node*>::iterator nearestGrandchildIt;
    DistanceType distanceNearestGrandchild =
        std::numeric_limits<DistanceType>::infinity();
    for (auto it = grandchildren.begin(); it != grandchildren.end(); it++) {
        DistanceType distance = distance_function((*it)->getKey(), to.getKey());
        if (distance < distanceNearestGrandchild) {
            distanceNearestGrandchild = distance;
            nearestGrandchildIt = it;
        }
    }
    Node& nearestGrandchild = **nearestGrandchildIt;
    bool recompute_donors_radius;
    if (from.is_leaf) {
        recompute_donors_radius = from.furthest_descendant == &nearestGrandchild;
        from.stats.removeDescendant(static_cast<ObjectNode&>(nearestGrandchild).value);
        to.stats.addDescendant(static_cast<ObjectNode&>(nearestGrandchild).value);
    }
    else {
        recompute_donors_radius =
            isDescendant(*from.furthest_descendant, static_cast<RoutingNode&>(nearestGrandchild));
        from.stats.removeDescendants(static_cast<RoutingNode&>(nearestGrandchild).stats);
        to.stats.addDescendants(static_cast<RoutingNode&>(nearestGrandchild).stats);
    }
    to.children.push_back(&nearestGrandchild);
    grandchildren.erase(nearestGrandchildIt);
    nearestGrandchild.parent = &to;
    nearestGrandchild.parent_distance = distanceNearestGrandchild;
    if (recompute_donors_radius)
        computeNodeRadius(from);
    if (!from.is_leaf) {
        RoutingNode& grandchild_as_RoutingNode = static_cast<RoutingNode&>(nearestGrandchild);
        if (distanceNearestGrandchild + grandchild_as_RoutingNode.covering_radius > lower_bound_factor * to.covering_radius)
            computeNonLeafRadius(to, grandchild_as_RoutingNode, true);
    }
    else if (to.covering_radius < nearestGrandchild.parent_distance) {
        to.furthest_descendant = static_cast<ObjectNode*>(&nearestGrandchild);
        to.covering_radius = nearestGrandchild.parent_distance;
    }
}

template<typename Key, typename T, typename DistanceType, typename NodeStats>
void MTree<Key,T,DistanceType,NodeStats>::mergeRoutingNodes(RoutingNode& from, RoutingNode& to, DistanceType from_to_distance) {
    assert (from.parent == to.parent);
    assert (from.is_leaf == to.is_leaf);
    if (!from.is_leaf) {
        if (from_to_distance + from.covering_radius > lower_bound_factor * to.covering_radius)
            computeNonLeafRadius(to, from, true);
    }
    for (Node* child : from.children) {
        child->parent = &to;
        child->parent_distance =
            distance_function (child->getKey(), to.getKey());
        if (from.is_leaf && child->parent_distance > to.covering_radius) {
            to.furthest_descendant = static_cast<ObjectNode*>(child);
            to.covering_radius = child->parent_distance;
        }
    }
    to.children.insert(to.children.end(), from.children.begin(), from.children.end());
    to.stats.addDescendants(from.stats);
    std::vector<Node*>& siblings = from.parent->children;
    auto to_erase_it =
        std::find(siblings.begin(), siblings.end(), &from);
    assert(to_erase_it != siblings.end());
    siblings.erase(to_erase_it);
    delete &from;
}

template<typename Key, typename T, typename DistanceType, typename NodeStats>
void MTree<Key,T,DistanceType,NodeStats>::rebalanceNode(RoutingNode& node) {
    assert (!node.parent->is_leaf);
    RoutingNode *nearestDonor = nullptr;
    DistanceType distanceNearestDonor = // TODO: change names
        std::numeric_limits<DistanceType>::infinity();
    RoutingNode *nearestMergeCandidate = nullptr;
    DistanceType distanceNearestMergeCandidate =
        std::numeric_limits<DistanceType>::infinity();

    for (Node* child : node.parent->children) {
        RoutingNode& child_as_RoutingNode = static_cast<RoutingNode&>(*child);
        assert(node.is_leaf == child_as_RoutingNode.is_leaf);
        if (child != &node) {
            DistanceType distance =
                distance_function(node.getKey(), child_as_RoutingNode.key);
            std::vector<Node*>& grandchildren = child_as_RoutingNode.children;
            if (grandchildren.size() > min_node_size) {
                if (distance < distanceNearestDonor) {
                    distanceNearestDonor = distance;
                    nearestDonor = &child_as_RoutingNode;
                }
            }
            else {
                if (distance < distanceNearestMergeCandidate) {
                    distanceNearestMergeCandidate = distance;
                    nearestMergeCandidate = &child_as_RoutingNode;
                }
            }
        }
    }
    if (nearestDonor != nullptr) {
        donateChild(*nearestDonor, node);
        assert (node.children.size() >= min_node_size);
    }
    else {
        assert (nearestMergeCandidate != nullptr);
        mergeRoutingNodes(node, *nearestMergeCandidate, distanceNearestMergeCandidate);
        assert(nearestMergeCandidate->children.size() >= min_node_size);
    }
}

template<typename Key, typename T, typename DistanceType, typename NodeStats>
void MTree<Key,T,DistanceType,NodeStats>::pullUpRoot() {
    // code for handling root.children.size() == 1
    if (entries.size() == 1) {
        // Clear tree
        assert (root->is_leaf);
        delete root;
        root = nullptr;
    }
    else if (!root->is_leaf) {
        assert (root->children.size() == 1);
        RoutingNode* old_root = root;
        root = static_cast<RoutingNode*>(root->children[0]);
        root->parent = nullptr;
        root->parent_distance = 0;
        // no need to keep track of covering radius of root
        root->covering_radius = 0;
        root->furthest_descendant = nullptr;
        delete old_root;
    }
}

template<typename Key, typename T, typename DistanceType, typename NodeStats>
void MTree<Key,T,DistanceType,NodeStats>::treeErase(ObjectNode& to_erase) {
    assert (root != nullptr);
    assert (!entries.empty());
    
    RoutingNode& parent = *to_erase.parent;
    auto to_erase_it = 
        std::find(parent.children.begin(), parent.children.end(), &to_erase);
    assert (to_erase_it != parent.children.end());
    parent.children.erase(to_erase_it);
    for (auto node = &parent; node != nullptr; node = node->parent) {
        if (node->furthest_descendant == &to_erase)
            computeNodeRadius(*node);
        assert(node->furthest_descendant != &to_erase);
        node->stats.removeDescendant(to_erase.value);
    }
    for (RoutingNode* node = &parent; node != root; ) {
        RoutingNode* node_parent = node->parent;
        if (node->children.size() >= min_node_size)
            break;
        rebalanceNode(*node);
        // *node might have been erased in rebalanceNode()
        node = node_parent;
    }
    assert(entries.size() > min_node_size);
    if (root->children.size() <= 1) {
        pullUpRoot();
    }
}

template<typename Key, typename T, typename DistanceType, typename NodeStats>
void MTree<Key,T,DistanceType,NodeStats>::erase(iterator pos) {
    treeErase(*static_cast<EntryListIterator>(pos));
    entries.erase(pos);
    assert(entries.empty() == (root == nullptr));
    check();
}

template<typename Key, typename T, typename DistanceType, typename NodeStats>
auto MTree<Key,T,DistanceType,NodeStats>::nnSubtreeSearch(const KeyType& needle, DistanceType min_radius, DistanceType max_radius, bool reverse, RoutingNode& subtree, bool locking) -> std::pair<iterator,DistanceType> {
    BoundEstimator bound_estimator;
    iterator best = end();
    DistanceType best_distance = reverse ? min_radius : max_radius;
    DistanceType best_bound = best_distance;
    if (reverse) {
        bound_estimator = [this,&best_bound](Query& query, const NodeStatsType*, NodeTagType, NodeTagType, DistanceType lower, DistanceType) {
            if (lower > best_bound) {
                best_bound = lower;
                query.setMinRadius(best_bound);
            }
        };
    }
    else {
        bound_estimator = [this,&best_bound](Query& query, const NodeStatsType*, NodeTagType, NodeTagType, DistanceType, DistanceType upper) {
            if (upper < best_bound) {
                best_bound = upper;
                query.setMaxRadius(best_bound);
            }
        };
    }
    Query query = subtreeSearch(needle, min_radius, max_radius, reverse, bound_estimator, subtree, locking);
    for (; !query.atEnd(); ++query) {
        iterator entry = (*query).first;
        DistanceType distance = (*query).second;
        assert(reverse ? (distance >= best_distance) : (distance <= best_distance));
        if (best_distance != distance || best == end()) {
            best_distance = distance;
            best = entry;
        }
    }
    return std::make_pair(best,best_distance);
}


template<typename Key, typename T, typename DistanceType, typename NodeStats>
template<typename TieBreaker>
auto MTree<Key,T,DistanceType,NodeStats>::knnSubtreeSearch(const KeyType& needle, unsigned k, bool sort, DistanceType min_radius, DistanceType max_radius, bool reverse, TieBreaker tie_breaker, bool extend_for_ties, RoutingNode& subtree) -> std::vector<std::pair<iterator,DistanceType>> {
    PlaceholderQueue<DistanceType,iterator,NodeTagType> queue;
    DistanceType max_distance;
    BoundEstimator bound_estimator;
    std::function<bool(const DistanceType&,const DistanceType&)> cmp;
    if (reverse) {
        cmp = std::greater<DistanceType>{};
        max_distance = std::numeric_limits<DistanceType>::lowest();
        bound_estimator = [k,&queue](Query& query, const NodeStatsType* stats, NodeTagType tag, NodeTagType parent, DistanceType lower, DistanceType) {
            if (lower > queue.getMaxKey()) {
                queue.addPlaceholder(parent, lower, (stats == nullptr) ? 1 : stats->getDescendantCount(), tag);
                if (queue.getMaxKey() > query.getMinRadius())
                    query.setMinRadius(queue.getMaxKey());
            }
        };
    }
    else {
        cmp = std::less<DistanceType>{};
        max_distance = std::numeric_limits<DistanceType>::max();
        bound_estimator = [k,&queue](Query& query, const NodeStatsType* stats, NodeTagType tag, NodeTagType parent, DistanceType, DistanceType upper) {
            if (upper < queue.getMaxKey()) {
                queue.addPlaceholder(parent, upper, (stats == nullptr) ? 1 : stats->getDescendantCount(), tag);
                if (queue.getMaxKey() < query.getMaxRadius())
                    // In rare situations (hash collisions), queue.getMaxKey() might increase
                    query.setMaxRadius(queue.getMaxKey());
            }
        };
    }
    std::vector<std::pair<iterator,DistanceType>> result;
    auto pair_cmp = [reverse,&tie_breaker](const typename decltype(result)::value_type& a, const typename decltype(result)::value_type& b) {
        if (a.second == b.second) {
            return tie_breaker(*a.first, *b.first);
        }
        else {
            return reverse ? (a.second > b.second) : (a.second < b.second);
        } 
    };
    queue = {k, cmp, max_distance};
    Query query = subtreeSearch(needle, min_radius, max_radius, reverse, bound_estimator, subtree, false);
    for (; !query.atEnd(); ++query) {
        auto item = *query;
        if (result.size() < k) {
            result.push_back(item);
            std::push_heap(result.begin(), result.end(), pair_cmp);
        }
        else if (item.second == result[0].second && extend_for_ties) {
            result.push_back(item);
            std::push_heap(result.begin(), result.end(), pair_cmp);
        }
        else if (pair_cmp(item, result[0])) {
            if (result.size() > k) {
                // In case of ties and extend_for_ties == true,
                // we might have more than k elements
                result.resize(k);
            }
            std::pop_heap(result.begin(), result.end(), pair_cmp);
            result[k-1] = item;
            std::push_heap(result.begin(), result.end(), pair_cmp);
        }
    }
    if (sort)
        std::sort_heap(result.begin(), result.end(), pair_cmp);
    return result;
}

template<typename Key, typename T, typename DistanceType, typename NodeStats>
template<typename KeySer, typename TSer>
void MTree<Key,T,DistanceType,NodeStats>::serialize(std::ostream& out, KeySer& key_ser, TSer& T_ser) {
    std::unordered_map<Node*,int> pointer_map;
    std::unordered_set<RoutingNode*> routing_nodes;
    std::size_t i = 0;
    pointer_map[root] = 0;
    for (auto& node : entries) {
        pointer_map[&node] = i++;
        RoutingNode* parent = node.parent;
        while (parent != nullptr && !routing_nodes.count(parent)) {
            routing_nodes.insert(parent);
            parent = parent->parent;
        }
    }
    for (RoutingNode* node : routing_nodes) {
        pointer_map[node] = i++;
    }
    serializeInt<std::uint64_t>(out, min_node_size);
    serializeInt<std::uint64_t>(out, max_node_size);
    serializeInt<std::uint64_t>(out, split_sampling);
    serializeInt<std::uint64_t>(out, pointer_map[root]);
    serializeInt<std::uint64_t>(out, entries.size());
    serializeInt<std::uint64_t>(out, routing_nodes.size());
    for (auto& node : entries) {
        serializeDistance<DistanceType>(out, node.parent_distance);
        key_ser(out, node.value.first);
        T_ser(out, node.value.second);
    }
    for (RoutingNode* node : routing_nodes) {
        serializeDistance<DistanceType>(out, node->parent_distance);
        serializeInt<std::uint8_t>(out, node->is_leaf);
        key_ser(out, node->key);
        node->stats.serialize(out);
        serializeDistance<DistanceType>(out, node->covering_radius);
        serializeInt<std::uint64_t>(out, pointer_map[node->furthest_descendant]);
        serializeInt<std::uint64_t>(out, node->children.size());
        for (Node *child : node->children)
            serializeInt<std::uint64_t>(out, pointer_map[child]);
    }
}

template<typename Key, typename T, typename DistanceType, typename NodeStats>
template<typename KeyUnser, typename TUnser>
void MTree<Key,T,DistanceType,NodeStats>::unserialize(std::istream& in, KeyUnser& key_unser, TUnser& T_unser) {
    clear();
    min_node_size = unserializeInt<std::uint64_t>(in);
    max_node_size = unserializeInt<std::uint64_t>(in);
    split_sampling = unserializeInt<std::uint64_t>(in);
    std::size_t root_index = unserializeInt<std::uint64_t>(in);
    std::size_t entry_cnt = unserializeInt<std::uint64_t>(in);
    std::size_t routing_node_cnt = unserializeInt<std::uint64_t>(in);
    std::vector<Node*> pointers;
    pointers.reserve(entry_cnt + routing_node_cnt);
    for (std::size_t i = 0; i < entry_cnt; i++) {
        DistanceType parent_distance = unserializeDistance<DistanceType>(in);
        KeyType key = key_unser(in);
        entries.emplace_back(std::make_pair(std::move(key), T_unser(in)));
        entries.back().parent_distance = parent_distance;
        entries.back().list_ref = std::prev(entries.end());
        pointers.push_back(&entries.back());
    }
    for (std::size_t i = 0; i < routing_node_cnt; i++) {
        pointers.push_back(new RoutingNode());
    }
    for (std::size_t i = 0; i < routing_node_cnt; i++) {
        RoutingNode& node = static_cast<RoutingNode&>(*pointers[entry_cnt+i]);
        node.parent_distance = unserializeDistance<DistanceType>(in);
        node.is_leaf = unserializeInt<uint8_t>(in);
        node.key = key_unser(in);
        node.stats.unserialize(in);
        node.covering_radius = unserializeDistance<DistanceType>(in);
        node.furthest_descendant =
            static_cast<ObjectNode*>(pointers[unserializeInt<std::uint64_t>(in)]);
        node.children.resize(unserializeInt<std::uint64_t>(in));
        for (Node*& child_ptr : node.children) {
            child_ptr = pointers[unserializeInt<std::uint64_t>(in)];
            child_ptr->parent = &node;
        }
    }
    root = routing_node_cnt == 0 ?
        nullptr :
        static_cast<RoutingNode*>(pointers[root_index]) ;
}

static inline bool might_be_contained(double distance, double parent_distance, double upper_bound) {
    // this does std::abs(distance - parent_distance) <= bound
    // avoid minus for numerical reasons
    return (distance > parent_distance) ?
        (distance <= upper_bound + parent_distance) :
        (parent_distance <= upper_bound + distance) ;
}


// TODO: RangeQuery code could probably be optimized
template<typename Key, typename T, typename DistanceType, typename NodeStats>
void MTree<Key,T,DistanceType,NodeStats>::RangeQuery::findNextLeaf() {
    while (!queue.empty()) {
        RoutingNode& node = *queue.top().first;
        DistanceType node_distance = queue.top().second;
        queue.pop();
        if (node.is_leaf) {
            current_leaf = &node;
            current_leaf_distance = node_distance;
            return;
        }
        for (Node* child : node.children) {
            RoutingNode& child_as_RoutingNode = static_cast<RoutingNode&>(*child);
            double upper_bound = upper_bound_factor * radius + child_as_RoutingNode.covering_radius;
            if (might_be_contained(node_distance, child->parent_distance, upper_bound)) {
                DistanceType distance = tree->distance_function(child_as_RoutingNode.key, needle);
                if (distance <= upper_bound_factor * radius + child_as_RoutingNode.covering_radius) {
                    queue.emplace(&child_as_RoutingNode, distance);
                }
            }
        }
    }
}

// TODO: maybe merge RangeQuery and Query. Being able to use a queue instead of a vector doesnt seem to justify a separate implementation
template<typename Key, typename T, typename DistanceType, typename NodeStats>
auto MTree<Key,T,DistanceType,NodeStats>::RangeQuery::operator++() -> RangeQuery& {
    for (;;) {
        if (current_leaf == nullptr) {
            findNextLeaf();
            if (current_leaf == nullptr)
                break;
            leaf_it = current_leaf->children.begin();    
        }
        while (leaf_it != current_leaf->children.end()) {
            ObjectNode& child = static_cast<ObjectNode&>(**leaf_it);
            leaf_it++;
            double upper_bound = upper_bound_factor * radius;
            if (might_be_contained(current_leaf_distance, child.parent_distance, upper_bound)) {
                DistanceType distance = tree->distance_function(child.getKey(), needle);
                // no upper_bound_factor here
                if (distance <= radius) {
                    last_returned = child.list_ref;
                    last_distance = distance;
                    return *this;
                }
            }
        }
        current_leaf = nullptr;
    }
    is_at_end = true;
    return *this;
}


// TODO:
// * maybe allow pruning min_search radius for reverse
// * try with a heap, simply ignoring pruning
// * try with a tree


template<typename Key, typename T, typename DistanceType, typename NodeStats>
bool MTree<Key,T,DistanceType,NodeStats>::Query::pruneQueue(RoutingNode** node, DistanceType* distance) {
    if (queue.empty())
        return true;
    std::pop_heap(queue.begin(), queue.end(), queue_cmp);
    bool done = reverse ?
        (queue.back().distance_bound < tolerant_min_search_radius) :
        (queue.back().distance_bound > tolerant_max_search_radius) ;
    if (done) {
        return true;
    }

    *node = queue.back().node;
    *distance = queue.back().center_distance;
    queue.pop_back();
    return false;
}

// template<typename Key, typename T, typename DistanceType, typename NodeStats>
// bool MTree<Key,T,DistanceType,NodeStats>::Query::pruneQueue(RoutingNode** node, DistanceType* distance) {
//     if (reverse) {
//         auto last = queue.lower_bound(lower_bound_factor * min_search_radius);
//         queue.erase(queue.begin(), last);
//         if (queue.empty())
//         // if (last == queue.end())
//             return true;
//         auto best = std::prev(queue.end());
//         *node = best->second.node;
//         *distance = best->second.center_distance;
//         queue.erase(best);
//     }
//     else {
//         auto first = queue.upper_bound(upper_bound_factor * max_search_radius);
//         queue.erase(first, queue.end());
//         if (queue.empty())
//         // if (first == queue.begin())
//             return true;
//         auto best = queue.begin();
//         *node = best->second.node;
//         *distance = best->second.center_distance;
//         queue.erase(best);
//     }
//     return false;
// }

// template<typename Key, typename T, typename DistanceType, typename NodeStats>
// bool MTree<Key,T,DistanceType,NodeStats>::Query::pruneQueue(RoutingNode** node, DistanceType* distance) {
//     std::vector<QueueEntry> new_queue;
//     new_queue.reserve(queue.capacity());
//     int best_entry = -1;
//     std::function<bool(DistanceType,DistanceType)> is_better;
//     if (reverse) {
//         is_better = std::greater<DistanceType>{};
//     }
//     else  {
//         is_better = std::less<DistanceType>{};
//     }
//     for (auto& entry : queue) {
//         bool prune = reverse ?
//             entry.distance_bound < lower_bound_factor * min_search_radius :
//             entry.distance_bound > upper_bound_factor * max_search_radius ;
//         if (!prune) {
//             new_queue.push_back(entry);
//             if (best_entry == -1 || is_better(entry.distance_bound, new_queue[best_entry].distance_bound))
//                 best_entry = new_queue.size()-1;
//         }
//     }
//     queue = std::move(new_queue);
//     //if (best_entry != (int)queue.size() - 1)
//     //    std::swap(queue[best_entry], queue[queue.size()-1]);

//     if (queue.empty())
//         return true;

//     *node = queue[best_entry].node;
//     *distance = queue[best_entry].center_distance;
//     if (best_entry != (int)queue.size() -1)
//         queue[best_entry] = queue[queue.size()-1];
//     queue.pop_back();
//     return false;
// } 

template<typename Key, typename T, typename DistanceType, typename NodeStats>
void MTree<Key,T,DistanceType,NodeStats>::Query::findNextLeaf() {
    for (;;) {
        RoutingNode* node;
        DistanceType distance;

        if (pruneQueue(&node, &distance)) {
            current_leaf = nullptr;
            return;
        }
        if (locking) {
            // For computeNodeRadius() being called during treeInsert
            // we need to make sure this branch isn't being rebuilt at the moment
            // TODO: make this conditional (and check)
            node->mutex.lock();
            node->mutex.unlock();
        }

        if (node->is_leaf) {
            current_leaf = node;
            current_leaf_distance = distance;
            return;
        }
        for (Node* child : node->children) {
            RoutingNode& child_as_RoutingNode = static_cast<RoutingNode&>(*child);
            if (distance + child->parent_distance + child_as_RoutingNode.covering_radius < tolerant_min_search_radius)
                continue;
            double upper_bound = tolerant_max_search_radius + child_as_RoutingNode.covering_radius;
            if (!might_be_contained(distance, child->parent_distance, upper_bound)) 
                continue;
            DistanceType child_distance = tree->distance_function(child_as_RoutingNode.key, needle);
            // distance of all descendants of child is >d_min
            DistanceType d_min = std::max<DistanceType>(child_distance - child_as_RoutingNode.covering_radius, 0);
            // distance of all descendants of child is <d_max
            DistanceType d_max = child_distance + child_as_RoutingNode.covering_radius;
            if (d_max < tolerant_min_search_radius)
                continue;
            if (d_min > tolerant_max_search_radius)
                continue;
            // queue.emplace(std::make_pair(reverse ? d_max : d_min, QueueEntry{&child_as_RoutingNode, child_distance, reverse ? d_max : d_min}));
            queue.emplace_back(&child_as_RoutingNode, child_distance, reverse ? d_max : d_min);
            std::push_heap(queue.begin(), queue.end(), queue_cmp);
            // We supply the bound estimator by a lower and upper bound of the
            // distances of all entries descending this node. It can make
            // use of it by setting a new min_search_radius or max_search_radius.
            bound_estimator(
                *this,
                &child_as_RoutingNode.stats,
                &child_as_RoutingNode,
                child_as_RoutingNode.parent,
                lower_bound_factor * d_min,
                upper_bound_factor * d_max
            );
        }
    }
}

template<typename Key, typename T, typename DistanceType, typename NodeStats>
auto MTree<Key,T,DistanceType,NodeStats>::Query::operator++() -> Query& {
    for (;;) {
        if (current_leaf == nullptr) {
            findNextLeaf();
            if (current_leaf == nullptr)
                break;
            leaf_it = current_leaf->children.begin();
        }
        while (leaf_it != current_leaf->children.end()) {
            ObjectNode& child = static_cast<ObjectNode&>(**leaf_it);
            leaf_it++;
            if (current_leaf_distance + child.parent_distance < tolerant_min_search_radius)
                continue;
            if (!might_be_contained(current_leaf_distance, child.parent_distance, tolerant_max_search_radius))
                continue;
            DistanceType child_distance = tree->distance_function(child.getKey(), needle);
            bound_estimator(*this, nullptr, NodeTagType{}, child.parent, child_distance, child_distance);
            // no lower_bound_factor / upper_bound_factor here
            if (child_distance >= min_search_radius && child_distance <= max_search_radius) {
                last_returned = child.list_ref;
                last_distance = child_distance;
                return *this;
            }
        }
        current_leaf = nullptr;
    }
    is_at_end = true;
    return *this;
}


