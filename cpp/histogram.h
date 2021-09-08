// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

#ifndef DSALMON_HISTOGRAM_H
#define DSALMON_HISTOGRAM_H

#include "Vector.h"

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/ranked_index.hpp>
#include <boost/multi_index/identity.hpp>

#include <atomic>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>

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