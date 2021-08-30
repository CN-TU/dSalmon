// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

#ifndef DSALMON_SLIDINGWINDOW_H
#define DSALMON_SLIDINGWINDOW_H

#include "MTree.h"
#include "Vector.h"

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/ranked_index.hpp>
#include <boost/multi_index/identity.hpp>

#include <map>
#include <atomic>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>


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


template<typename FloatType=double>
class SWHBOS {
	// we use an order statistic tree provided by boost to
	// retain histograms while adding and removing points
	class Histogram {

		typedef boost::multi_index_container<
			FloatType,
			boost::multi_index::indexed_by<boost::multi_index::ranked_non_unique<boost::multi_index::identity<FloatType>>>
		> MapType;

		MapType map;

	  public:
	  	typedef typename MapType::template nth_index<0>::type::iterator iterator;

	  	iterator add(FloatType value) {
			  auto inserted = map.template get<0>().insert(value);
			  return inserted.first;
		}

		void erase(iterator pos) {
			map.template get<0>().erase(pos);
		}

		FloatType getNormBinHeight(std::size_t n_bins, iterator pos) {
			auto& index = map.template get<0>();
			if (*index.begin() == *std::prev(index.end()))
				// edge case: all entries have the same value
				return 1;
			// how many samples we put in one bin
			const std::size_t per_bin =
				std::max<std::size_t>(1,index.size() / n_bins);
			const FloatType value = *pos;
			// height of current bin, max. of all bin heights
			FloatType bin_height = 0, max_bin_height = 0;
			// Indices of beginning and end of samples in current bin
			std::size_t start = 0, stop = 0;
			// lower boundary of values belonging to the current bin
			FloatType lower_boundary = *index.begin();
			// upper boundary of values belonging to the current bin
			FloatType upper_boundary;
			while (stop < index.size()) {
				stop = start + per_bin;
				if (stop >= index.size()) {
					stop = index.size();
					upper_boundary = *std::prev(index.end());
				}
				else {
					auto stopIt = index.nth(stop);
					if (*std::prev(stopIt) == *stopIt) {
						// move stopIt to the end of a range of
						// equal-value entries
						stopIt = index.upper_bound(*stopIt);
					}
					if (stopIt == index.end()) {
						stop = index.size();
						upper_boundary = *std::prev(index.end());
					}
					else {
						stop = index.rank(stopIt);
						upper_boundary = (*stopIt + *std::prev(stopIt)) / 2;
					}
				}
				FloatType this_bin_height = (stop - start) / (upper_boundary - lower_boundary);
				max_bin_height = std::max(this_bin_height, max_bin_height);
				if (lower_boundary <= value && upper_boundary >= value)
					// this is the bin of the currently processed sample
					bin_height = this_bin_height;
				lower_boundary = upper_boundary;
				start = stop; 
			}
			return bin_height / max_bin_height;
		}
	};

	struct Point {
		Vector<FloatType> data;
		FloatType expire_time;
		std::vector<typename Histogram::iterator> iterators;
	};
	// store processed points in the order of their arrival
	typedef std::list<Point> PointList;
	typedef typename PointList::iterator PointListIterator;

	class ThreadPool {
		SWHBOS<FloatType>& detector;
		std::mutex mutex, done_mutex;
		std::condition_variable cv;
		bool stop;
		std::size_t jobs_running;
		std::vector<std::thread> workers;

		PointListIterator score_item;
		// std::size_t current_histogram;
		std::atomic<unsigned> current_histogram;
		FloatType outlierness_score;

		void worker() {
			std::unique_lock<std::mutex> lock(mutex);
			for (;;) {
				if (stop)
					break;
				if (score_item != detector.points.end()) {
					unsigned i;
					FloatType local_score = 0;
					jobs_running++;
					lock.unlock();
					while ((i = current_histogram++) < detector.histograms.size()) {
						FloatType bin_height = detector.histograms[i].getNormBinHeight(detector.n_bins, score_item->iterators[i]);
						local_score -= std::log10(bin_height);
					}
					lock.lock();
					outlierness_score += local_score;
					if (!--jobs_running) {
						score_item = detector.points.end();
						done_mutex.unlock();
					}
				}
				cv.wait(lock);
			}
		}

	  public:
		ThreadPool(SWHBOS<FloatType>& detector, std::size_t n_threads) :
			detector(detector), 
			stop(false),
			jobs_running(0),
			score_item(detector.points.end())
		{
			done_mutex.lock();
			for (std::size_t i = 0; i < n_threads; i++)
				workers.emplace_back(&ThreadPool::worker, &*this);
		}

		~ThreadPool() {
			mutex.lock();
			stop = true;
			mutex.unlock();
			cv.notify_all();
			for (auto& thread : workers)
				thread.join();
		}

		FloatType outlierness(PointListIterator pos) {
			mutex.lock();
			score_item = pos;
			current_histogram = 0;
			outlierness_score = 0;
			mutex.unlock();
			cv.notify_all();
			done_mutex.lock();
			return outlierness_score;
		}
	};

	// size of the sliding window
	FloatType window;
	// how many bins per histogram we use
	std::size_t n_bins;

	PointList points;

	// one histogram per data dimension
	std::vector<Histogram> histograms;
	std::unique_ptr<ThreadPool> thread_pool;

	void eraseFromHistograms(Point& point) {
		std::vector<typename Histogram::iterator>& iterators = point.iterators;
		for (std::size_t i = 0; i < histograms.size(); i++)
			histograms[i].erase(iterators[i]);
	}

	void pruneExpired(FloatType now) {
		while (!points.empty() && points.front().expire_time <= now) {
			eraseFromHistograms(points.front());
			points.pop_front();
		}
	}

  public:
	SWHBOS(FloatType window, std::size_t n_bins) :
		window(window),
		n_bins(n_bins)
	{ }

	void initThreadPool(std::size_t n_threads) {
		thread_pool.reset(new ThreadPool(*this, n_threads));
	}

	void releaseThreadPool() {
		thread_pool.reset();
	}

	std::size_t windowSize() { return points.size(); }

	class WindowSample{
		PointListIterator it;
	public:
		WindowSample(PointListIterator it) : it(it) {}
		Vector<FloatType> getSample() { return it->data; }
		FloatType getExpireTime() { return it->expire_time; }
	};
	
	class iterator : public PointListIterator {
	  public:
		WindowSample operator*() { return WindowSample(PointListIterator(*this)); };
		iterator() {}
		iterator(PointListIterator it) : PointListIterator(it) {}
	};

	iterator begin() { return iterator(points.begin()); }
	iterator end() { return iterator(points.end()); }

	iterator append(const Vector<FloatType>& data, FloatType now) {
		if (histograms.empty())
			histograms.resize(data.size());

		pruneExpired(now);

		auto it = points.emplace(points.end());
		it->expire_time = now + window;
		it->iterators.reserve(histograms.size());
		for (std::size_t i = 0; i < histograms.size(); i++)
			it->iterators[i] = histograms[i].add(data[i]);
		it->data = std::move(data);
		return iterator(it);
	}
	
	FloatType outlierness(iterator pos) {
		// compute outlierness score according to HBOS paper
		if (thread_pool) {
			return thread_pool->outlierness(pos);
		}
		else {
			FloatType score = 0;
			for (std::size_t i = 0; i < histograms.size(); i++) {
				FloatType bin_height = histograms[i].getNormBinHeight(n_bins, pos->iterators[i]);
				score -= std::log10(bin_height);
			}
			return score;
		}
	}
};

#endif
