// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

#ifndef DSALMON_HSTREE_H
#define DSALMON_HSTREE_H

#include <limits>
#include <vector>

#include "Vector.h"

template<typename FloatType=double>
class HSTrees {
    struct Node {
        int q;
        FloatType p;
        std::size_t l;
        std::size_t r;
    };

    class Tree {
        std::vector<Node> nodes;
        int tree_depth;
      public:
        Tree(std::size_t tree_depth) : nodes( (1<<(tree_depth+1)) - 1), tree_depth(tree_depth) {}
        Node& getRoot() { return nodes[0]; };
        Node& getLeftChild(Node& node, int node_depth) {
            return *(&node + 1);
        }
        Node& getRightChild(Node& node, int node_depth) {
            return *(&node + (1 << (tree_depth-node_depth)));
        }
        void updateModel() {
            for (Node& node : nodes) {
                node.r = node.l;
                node.l = 0;
            }
        }
    };

    std::vector<Tree> trees;
    FloatType window;
    int max_depth;
    std::size_t size_limit;

    int dimension;
    std::mt19937 rng;
    bool initial_model_completed;
    FloatType last_model_update;

    void initTreeNode(Tree& tree, Node& node, int depth, const std::vector<FloatType> mins, const std::vector<FloatType> maxs) {
        if (depth == max_depth) {
            node.l = node.r = 0;
        }
        else {
            node.q = std::uniform_int_distribution<int>{0,dimension-1}(rng);
            node.p = std::uniform_real_distribution<FloatType>{mins[node.q], maxs[node.q]}(rng);
            node.l = node.r = 0;
            std::vector<FloatType> new_maxs = maxs;
            new_maxs[node.q] = node.p;
            initTreeNode(tree, tree.getLeftChild(node, depth), depth+1, mins, new_maxs);
            std::vector<FloatType> new_mins = mins;
            new_mins[node.q] = node.p;
            initTreeNode(tree, tree.getRightChild(node, depth), depth+1, new_mins, maxs);
        }
    }

    void initTree(Tree& tree) {
        std::uniform_real_distribution<FloatType> distribution{0,1};
        std::vector<FloatType> mins(dimension), maxs(dimension);

        for (int i = 0; i < dimension; i++) {
            FloatType s_q = distribution(rng);
            mins[i] = s_q - 2*std::max(s_q, 1-s_q);
            maxs[i] = s_q + 2*std::max(s_q, 1-s_q);
        }
        initTreeNode(tree, tree.getRoot(), 0, mins, maxs);
    }

  public:
    HSTrees(FloatType window, std::size_t tree_count, int max_depth, std::size_t size_limit, int seed) :
        trees(tree_count, Tree{max_depth}),
        window(window),
        max_depth(max_depth),
        size_limit(size_limit),
        dimension(-1),
        rng(seed),
        initial_model_completed(false)
    { }

    FloatType fitPredict(const Vector<FloatType>& data, FloatType now) {
        FloatType score = 0;
        if (dimension == -1) {
            dimension = data.size();
            for (auto& tree : trees)
                initTree(tree);
            last_model_update = now;
        }
        if (last_model_update + window <= now) {
            for (auto& tree : trees)
                tree.updateModel();
            initial_model_completed = true;
            last_model_update = now;
        }
        for (auto& tree : trees) {
            bool found_score_for_tree = false;
            Node *node = &tree.getRoot();
            for (int depth = 0; ; depth++) {
                if (!found_score_for_tree && (depth == max_depth || node->r <= size_limit)) {
                    score += node->r * (1<<depth);
                    found_score_for_tree = true;
                }
                node->l++;
                if (depth == max_depth)
                    break;
                node = (data[node->q] < node->p) ? &tree.getLeftChild(*node, depth) : &tree.getRightChild(*node, depth);
            }
        }
        return initial_model_completed ? score : std::numeric_limits<FloatType>::quiet_NaN();
    }
};

#endif
