// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

#ifndef DSALMON_RRCT_H
#define DSALMON_RRCT_H

#include <memory>
#include <list>
#include <vector>

#include "Vector.h"

template<typename FloatType, typename T>
class RRCT {

    typedef Vector<FloatType> VectorType;
    // since an RRCT is usually used within a forest, use a shared pointer for vector reference for efficiency
    // TODO: constness
    typedef std::pair<std::shared_ptr<VectorType>, T> ValueType;

    struct Leaf;
    struct Branch;

    struct BboxRef {
        const VectorType* lower;
        const VectorType* upper;
        BboxRef(const VectorType& lower, const VectorType& upper) : lower(&lower), upper(&upper) {}
    };
    struct Point {
        Leaf* leaf;
        ValueType value;
        typename std::list<Point>::iterator list_ref;
        typename std::list<Point*>::iterator leaf_list_ref;
    };
    struct Node {
        Branch *parent;
        virtual int getChildrenCount() = 0;
        virtual BboxRef getBboxRef() = 0;
    };
    struct Branch final : public Node {
        int children_cnt;
        Node* left_child;
        Node* right_child;
        int cut_dimension;
        FloatType cut_value;
        struct {
            VectorType lower;
            VectorType upper;
        } bbox;
        int getChildrenCount() override { return children_cnt; }
        BboxRef getBboxRef() override { return BboxRef{bbox.lower, bbox.upper}; }
    };
    struct Leaf final : public Node {
        std::list<Point*> points; // TODO: would a vector be better here ?
        int getChildrenCount() override { return points.size(); }
        BboxRef getBboxRef() override { return BboxRef{*points.front()->value.first, *points.front()->value.first}; }
    };

    typedef std::list<Point> PointList;
    typedef typename PointList::iterator PointListIterator;
    PointList points;

    Node* root;
    int dimension;
    std::mt19937 rng;

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

  private:
    void extendParentBboxes(Node *node) {
        BboxRef bbox = node->getBboxRef();
        Branch* parent = node->parent;
        while (parent != nullptr) {
            bool nothing_changed = true;
            for (int i = 0; i < dimension; i++) {
                bool update_lower = parent->bbox.lower[i] > (*bbox.lower)[i];
                bool update_upper = parent->bbox.upper[i] < (*bbox.upper)[i];
                if (update_lower)
                    parent->bbox.lower[i] = (*bbox.lower)[i];
                if (update_upper)
                    parent->bbox.upper[i] = (*bbox.upper)[i];
                nothing_changed &= !update_lower && !update_upper;
            }
            if (nothing_changed) {
                break;
            }
            parent = parent->parent;
        }
    }

    void tightenParentBboxes(Node *node) {
        while (node->parent != nullptr) {
            Branch* parent = node->parent;
            Node* sibling = parent->left_child == node ?
                parent->right_child :
                parent->left_child ;
            BboxRef bbox1 = node->getBboxRef();
            BboxRef bbox2 = sibling->getBboxRef();
            bool nothing_changed = true;
            for (int i = 0; i < dimension; i++) {
                FloatType lower = std::min((*bbox1.lower)[i], (*bbox2.lower)[i]);
                FloatType upper = std::max((*bbox1.upper)[i], (*bbox2.upper)[i]);
                bool update_lower = parent->bbox.lower[i] < lower;
                bool update_upper = parent->bbox.upper[i] > upper;
                if (update_lower)
                    parent->bbox.lower[i] = lower;
                if (update_upper)
                    parent->bbox.upper[i] = upper;
                assert(parent->bbox.lower != parent->bbox.upper);
                nothing_changed &= !update_lower && !update_upper;
            }
            if (nothing_changed) {
                break;
            }
            node = parent;
        }
    }

    std::tuple<int,bool,FloatType> insertPointCut(const VectorType& point, const BboxRef& bbox) {
        FloatType ranges[dimension]; // TODO: allocate on heap?
        FloatType tot_sum = 0;
        for (int i = 0; i < dimension; i++) {
            ranges[i] =
                std::max(point[i], (*bbox.upper)[i]) - 
                std::min(point[i], (*bbox.lower)[i]);
            tot_sum += ranges[i];
        }
        if (tot_sum == 0) {
            return std::make_tuple(0, true, point[0]);
        }
        FloatType r = std::uniform_real_distribution<FloatType>{0,tot_sum}(rng);
        FloatType partial_sum = ranges[0];
        int cut_dim;
        for (cut_dim = 0; cut_dim < dimension-1 && partial_sum <= r; cut_dim++)
            partial_sum += ranges[cut_dim+1];
        assert(ranges[cut_dim] > 0);
        return std::make_tuple(
            cut_dim, // cut dimension
            false, // point is not identical
            std::max(point[cut_dim], (*bbox.upper)[cut_dim]) + (r - partial_sum) // cut value
        );
    }

    iterator createPointLeafAndBranch(iterator pos, Node* node, bool is_right, int cut_dimension, FloatType cut_value, ValueType&& value) {
        Branch* new_branch = new Branch{};
        Leaf* new_leaf = new Leaf{};
        assert(node != nullptr);
        Branch* parent = node->parent;
        new_branch->parent = parent;
        if (is_right) {
                new_branch->left_child = node;
                new_branch->right_child = new_leaf;
        }
        else {
                new_branch->left_child = new_leaf;
                new_branch->right_child = node;
        }
        new_branch->children_cnt = node->getChildrenCount() + 1;
        new_branch->cut_dimension = cut_dimension;
        new_branch->cut_value = cut_value;
        BboxRef node_bbox = node->getBboxRef();
        new_branch->bbox.lower = *node_bbox.lower;
        new_branch->bbox.upper = *node_bbox.upper;
        node->parent = new_branch;
        if (parent == nullptr)
            root = new_branch;
        else if (parent->left_child == node)
            parent->left_child = new_branch;
        else
            parent->right_child = new_branch;
        new_leaf->parent = new_branch;
        auto it = points.emplace(static_cast<PointListIterator>(pos));
        it->list_ref = it;
        it->value = std::move(value);
        it->leaf = new_leaf;
        new_leaf->points.push_front(&*it);
        it->leaf_list_ref = new_leaf->points.begin();
        extendParentBboxes(new_leaf);
        return iterator(it);
    }

    Leaf* findLeaf(const VectorType& vec) {
        Node* node = root;
        for(;;) {
            Branch* branch = dynamic_cast<Branch*>(node);
            if (branch == nullptr) {
                Leaf* leaf = static_cast<Leaf*>(node);
                return leaf->points[0]->value.first == vec ?
                    leaf :
                    nullptr ;
            }
            BboxRef bbox = branch->getBboxRef();
            int cut_dim = branch->cut_dimension;
            node = vec[cut_dim] <= branch->cut_value ?
                branch->left_child :
                branch->right_child ;
            assert(node != nullptr);
        }
    }

    bool isCutPossible(const VectorType& point, const BboxRef& bbox, int cut_dimension, FloatType cut_value) {
        if ((cut_value < (*bbox.lower)[cut_dimension]) && (point[cut_dimension] <= cut_value))
            return true;
        if ((cut_value >= (*bbox.upper)[cut_dimension]) && (point[cut_dimension] > cut_value))
            return true;
        return false;
    }

    iterator treeInsert(iterator pos, ValueType&& value) {
        const VectorType& point = *value.first;
        Node* node = root;
        int cut_dimension;
        FloatType cut_value;
        for (;;) {
            bool identical;
            BboxRef bbox = node->getBboxRef();
            if(dynamic_cast<Branch*>(node) != nullptr) {
                assert(bbox.lower != bbox.upper);
            }
            std::tie(cut_dimension, identical, cut_value) = insertPointCut(*value.first, bbox);
            if (!identical && !isCutPossible(point, bbox, cut_dimension, cut_value)) {
                // This might be triggered for points that are almost 
                // identical due to numerical issues.
                if (dynamic_cast<Leaf*>(node) != nullptr) {
                    // for leafs make sure that a cut is possible
                    if (point[cut_dimension] > (*bbox.upper)[cut_dimension])
                        cut_value = (*bbox.upper)[cut_dimension];
                    else if (point[cut_dimension] < (*bbox.lower)[cut_dimension])
                        cut_value = point[cut_dimension];
                    else {
                        assert (false);
                    }
                }
            }
            if (identical) {
                // this is a duplicate of node
                // Branches cannot have bbox.lower == bbox.upper => this must be a Leaf
                assert(dynamic_cast<Leaf*>(node) != nullptr);
                Leaf* leaf = static_cast<Leaf*>(node);
                auto it = points.emplace(static_cast<PointListIterator>(pos));
                it->list_ref = it;
                it->value = std::move(value);
                it->leaf = leaf;
                leaf->points.push_front(&*it);
                it->leaf_list_ref = leaf->points.begin();
                return iterator(it);
            }
            else if (cut_value < (*bbox.lower)[cut_dimension]) {
                return createPointLeafAndBranch(pos, node, false,
                                                cut_dimension, cut_value,
                                                std::move(value));
            }
            else if (cut_value >= (*bbox.upper)[cut_dimension] && cut_value < point[cut_dimension]) {
                return createPointLeafAndBranch(pos, node, true,
                                                cut_dimension, cut_value,
                                                std::move(value));
            }
            else {
                assert (dynamic_cast<Branch*>(node) != nullptr);
                Branch* branch = static_cast<Branch*>(node);
                branch->children_cnt++;
                node = point[branch->cut_dimension] <= branch->cut_value ?
                    branch->left_child :
                    branch->right_child ;
            }
        }

    }

  public:
    RRCT(int seed) : root(nullptr), dimension(-1), rng(seed) {}

    RRCT(const RRCT& other) = delete;
    RRCT& operator=(const RRCT&) = delete;

    RRCT(RRCT&&) = default;
    RRCT& operator=(RRCT&&) = default;

    ~RRCT() {
        clear();
    }

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

    iterator insert(iterator pos, ValueType&& value) {
        if (empty()) {
            dimension = value.first->size();
            Leaf* leaf = new Leaf{};
            auto it = points.emplace(points.begin());
            leaf->parent = nullptr;
            leaf->points.push_front(&*it);
            it->leaf_list_ref = leaf->points.begin();
            it->list_ref = it;
            it->value = std::move(value);
            it->leaf = leaf;
            root = leaf;
            return iterator(it);
        }
        else {
            assert (value.first->size() == dimension);
            return treeInsert(pos, std::move(value));
        }
    }
    void push_back(ValueType&& value) { insert(end(), std::move(value)); }
    void push_front(ValueType&& value) { insert(begin(), std::move(value)); }

    void erase(iterator pos) {
        auto it = static_cast<PointListIterator>(pos);
        Leaf* leaf = it->leaf;
        if (size() == 1) {
            assert(leaf == root);
            clear();
            return;
        }
        leaf->points.erase(it->leaf_list_ref);
        Branch* parent = leaf->parent;
        if (leaf->points.empty()) {
            // The only possibility where parent == nullptr is if size() == 1.
            // This already has been handled
            assert(parent != nullptr);
            Node* sibling = parent->left_child == leaf ?
                parent->right_child : 
                parent->left_child ;
            Branch* grandparent = parent->parent;
            if (grandparent == nullptr)
                root = sibling;
            else if (grandparent->left_child == parent)
                grandparent->left_child = sibling;
            else
                grandparent->right_child = sibling;
            sibling->parent = grandparent;
            delete parent;
            delete leaf;
            tightenParentBboxes(sibling);
            parent = sibling->parent;
        }
        for (; parent != nullptr; parent = parent->parent) {
            parent->children_cnt--;
        }
        points.erase(it->list_ref);
    }

    void pop_back() { erase(std::prev(end())); }
    void pop_front() { erase(begin()); }

    void reverse() { points.reverse(); }

    void clear() {
        Node *node = root;
        while (node != nullptr) {
            Branch* branch = dynamic_cast<Branch*>(node);
            if (branch == nullptr) {
                // node is a Leaf
                Leaf* leaf = static_cast<Leaf*>(node);
                node = node->parent;
                delete leaf;
            }
            else {
                // node is a Branch
                if (branch->left_child != nullptr) {
                    node = branch->left_child;
                    branch->left_child = nullptr;
                }
                else if (branch->right_child != nullptr) {
                    node = branch->right_child;
                    branch->right_child = nullptr;
                }
                else {
                    node = branch->parent;
                    delete branch;
                }
            }
        }
        points.clear();
        root = nullptr;
        dimension = -1;
        //TODO: reset rng ?
    }

    FloatType codisp(iterator pos) {
        PointListIterator it = static_cast<PointListIterator>(pos);
        Leaf *leaf = it->leaf;
        assert(leaf != nullptr);

        if (leaf == root) {
            // all points in the tree are equal
            assert (leaf->parent == nullptr);
            return 0;
        }
        Branch *branch = leaf->parent;
        assert(branch != nullptr);
        FloatType co_displacement = (FloatType)(branch->children_cnt - leaf->points.size()) / leaf->points.size();

        while(branch->parent) {
            Branch* parent = branch->parent;
            co_displacement = std::max(
                co_displacement,
                (FloatType)(parent->children_cnt - branch->children_cnt) / branch->children_cnt
            );
            branch = parent;
        }
        return co_displacement;
    }
};

#endif
