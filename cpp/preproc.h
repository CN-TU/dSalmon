// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

#ifndef DSALMON_PREPROC_H
#define DSALMON_PREPROC_H

#include <cmath>
#include <list>

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/ranked_index.hpp>
#include <boost/multi_index/identity.hpp>

#include "Vector.h"

template<typename FloatType=double>
class SWZScoreScaler {

	struct StoredPoint {
		Vector<FloatType> data;
		Vector<FloatType> squared_data;
		FloatType expire_time;
	};

	struct Block {
		std::vector<StoredPoint> points;
		int pruned_from;
		Vector<FloatType> sum;
		Vector<FloatType> squares_sum;
	};

	const std::size_t MAX_BLOCK = 100;

	FloatType window;
	std::list<Block> sliding_window;
	std::size_t total_points_in_window;

	void pruneExpired(FloatType now) {
		while (!sliding_window.empty()) {
			Block &first_block = sliding_window.front();
			while (first_block.pruned_from < first_block.points.size()) {
				if (first_block.points[first_block.pruned_from].expire_time > now)
					return;
				first_block.pruned_from++;
				total_points_in_window--;
			}
			sliding_window.pop_front();
		}
	}

	void addPoint(const Vector<FloatType>& data, FloatType now) {
		Vector<FloatType> squared_data(data.size());
		for (std::size_t i = 0; i < data.size(); i++)
			squared_data[i] = data[i] * data[i];
		if (sliding_window.empty() || sliding_window.back().points.size() >= MAX_BLOCK) {
			sliding_window.emplace_back();
			std::size_t dimension = data.size();
			sliding_window.back().sum.resize(dimension);
			sliding_window.back().squares_sum.resize(dimension);
		}
		Block& last_block = sliding_window.back();
		StoredPoint point = { data, squared_data, now + window };
		last_block.points.push_back(point);
		last_block.sum += data;
		last_block.squares_sum += squared_data;
		total_points_in_window++;
	}

  public:
	SWZScoreScaler(FloatType window) :
		window(window),
		total_points_in_window(0)
	{ }

	void getMeans(Vector<FloatType>& total_mean, Vector<FloatType>& total_squares_mean) {
		total_mean.clear();
		total_squares_mean.clear();
		if (total_points_in_window == 0)
			return;
		std::size_t dimension = sliding_window.front().sum.size();
		total_mean.resize(dimension);
		total_squares_mean.resize(dimension);
		for (const auto& block : sliding_window) {
			if (block.pruned_from == 0) {
				total_mean += block.sum;
				total_squares_mean += block.squares_sum;
			}
			else {
				for (std::size_t i = block.pruned_from; i < block.points.size(); i++) {
					total_mean += block.points[i].data;
					total_squares_mean += block.points[i].squared_data;
				}
			}
		}
		for (std::size_t i = 0; i < total_mean.size(); i++) {
			total_mean[i] /= total_points_in_window;
			total_squares_mean[i] /= total_points_in_window;
		}
	}

	Vector<FloatType> transform(const Vector<FloatType> &data, FloatType now) {
		Vector<FloatType> total_mean, total_squares_mean;
		Vector<FloatType> normalized(data.size());
		pruneExpired(now);
		addPoint(data, now);
		getMeans(total_mean, total_squares_mean);
		for (std::size_t i = 0; i < data.size(); i++) {
			normalized[i] = (data[i] - total_mean[i]) / std::sqrt(total_squares_mean[i] - total_mean[i]*total_mean[i]);
		}
		return normalized;
	}
};

template<typename FloatType=double>
class SWQuantileScaler {
	typedef boost::multi_index_container<
		FloatType,
		boost::multi_index::indexed_by<boost::multi_index::ranked_non_unique<boost::multi_index::identity<FloatType>>>
	> TreeType;

	struct StoredPoint {
		std::vector<typename TreeType::template nth_index<0>::type::iterator> iterators;
		FloatType expire_time;
	};

	FloatType window;
	FloatType quantile;
	std::vector<TreeType> trees;
	std::list<StoredPoint> sliding_window;

	void pruneExpired(FloatType now) {
		while (!sliding_window.empty() && sliding_window.front().expire_time <= now) {
			auto& iterators = sliding_window.front().iterators;
			for (std::size_t i = 0; i < iterators.size(); i++)
				trees[i].template get<0>().erase(iterators[i]);
			sliding_window.pop_front();
		}
	}

	void addPoint(const Vector<FloatType>& data, FloatType now) {
		if (trees.empty())
			trees.resize(data.size());
		sliding_window.emplace_back();
		StoredPoint& new_entry = sliding_window.back();
		new_entry.iterators.resize(data.size());
		for (std::size_t i = 0; i < data.size(); i++) {
			auto inserted = trees[i].template get<0>().insert(data[i]);
			new_entry.iterators[i] = inserted.first;
		}
		new_entry.expire_time = now + window;
	}

public:
	SWQuantileScaler(FloatType window, FloatType quantile) :
		window(window),
		quantile(quantile)
	{ }

	Vector<FloatType> transform(const Vector<FloatType>& data, FloatType now) {
		Vector<FloatType> normalized(data.size());
		pruneExpired(now);
		addPoint(data, now);
		std::size_t max_n = sliding_window.size()-1;
		for (std::size_t i = 0; i < data.size(); i++) {
			auto& tree_index = trees[i].template get<0>();
			FloatType norm_lower =
				*(tree_index.nth(static_cast<std::size_t>(std::floor(max_n * quantile))));
			FloatType norm_upper =
				*(tree_index.nth(static_cast<std::size_t>(std::ceil(max_n * (1 - quantile)))));
			normalized[i] = (data[i] - norm_lower) / (norm_upper - norm_lower);
		}
		return normalized;
	}
};

#endif
