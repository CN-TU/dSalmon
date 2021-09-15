// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

#ifndef DSALMON_HSTREE_H
#define DSALMON_HSTREE_H

#include <limits>
#include <random>
#include <vector>

#include "Vector.h"

template<typename FloatType=double>
class HSTree {
    struct Node {
        int q;
        FloatType p;
        std::size_t l;
        std::size_t r;
    };

    std::vector<Node> nodes;

    FloatType window;
    int max_depth;
    std::size_t size_limit;

    int dimension;
    std::mt19937 rng;
    bool initial_model_completed;
    FloatType last_model_update;

    Node& getRoot() { return nodes[0]; };
    Node& getLeftChild(Node& node, int node_depth) {
        return *(&node + 1);
    }
    Node& getRightChild(Node& node, int node_depth) {
        return *(&node + (1 << (max_depth-node_depth)));
    }
    void updateModel() {
        for (Node& node : nodes) {
            node.r = node.l;
            node.l = 0;
        }
    }

    void initTreeNode(Node& node, int depth, const std::vector<FloatType>& mins, const std::vector<FloatType>& maxs) {
        if (depth == max_depth) {
            node.l = node.r = 0;
        }
        else {
            node.q = std::uniform_int_distribution<int>{0,dimension-1}(rng);
            node.p = std::uniform_real_distribution<FloatType>{mins[node.q], maxs[node.q]}(rng);
            node.l = node.r = 0;
            std::vector<FloatType> new_maxs = maxs;
            new_maxs[node.q] = node.p;
            initTreeNode(getLeftChild(node, depth), depth+1, mins, new_maxs);
            std::vector<FloatType> new_mins = mins;
            new_mins[node.q] = node.p;
            initTreeNode(getRightChild(node, depth), depth+1, new_mins, maxs);
        }
    }

    void initTree() {
        std::uniform_real_distribution<FloatType> distribution{0,1};
        std::vector<FloatType> mins(dimension), maxs(dimension);

        for (int i = 0; i < dimension; i++) {
            FloatType s_q = distribution(rng);
            mins[i] = s_q - 2*std::max(s_q, 1-s_q);
            maxs[i] = s_q + 2*std::max(s_q, 1-s_q);
        }
        initTreeNode(getRoot(), 0, mins, maxs);
    }

  public:
    HSTree(FloatType window, int max_depth, std::size_t size_limit, int seed) :
        nodes((1<<(max_depth+1)) - 1),
        window(window),
        max_depth(max_depth),
        size_limit(size_limit),
        dimension(-1),
        rng(seed),
        initial_model_completed(false)
    { }

    FloatType fitPredict(const Vector<FloatType>& data, FloatType now) {
        FloatType score = -1;
        if (dimension == -1) {
            dimension = data.size();
            initTree();
            last_model_update = now;
        }
        if (last_model_update + window <= now) {
            updateModel();
            initial_model_completed = true;
            last_model_update = now;
        }
        Node *node = &getRoot();
        for (int depth = 0; ; depth++) {
            if (score < 0 && (depth == max_depth || node->r <= size_limit))
                score = node->r * (1<<depth);
            node->l++;
            if (depth == max_depth)
                break;
            node = (data[node->q] < node->p) ? &getLeftChild(*node, depth) : &getRightChild(*node, depth);
        }
        return initial_model_completed ? -score : std::numeric_limits<FloatType>::quiet_NaN();
    }
};

#endif
