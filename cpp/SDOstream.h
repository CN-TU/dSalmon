// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

#ifndef DSALMON_SDOSTREAM_H
#define DSALMON_SDOSTREAM_H

#include <algorithm>
#include <boost/container/set.hpp>
#include <cmath>
#include <functional>
#include <limits>
#include <random>

#include "Vector.h"

template<typename FloatType=double>
class SDOstream {
  public:
    typedef std::function<FloatType(const Vector<FloatType>&, const Vector<FloatType>&)> DistanceFunction;

  private:
    // number of observers we want
    std::size_t observer_cnt;
    // fraction of observers to consider active
    FloatType active_observers;
    // factor for deciding if a sample should be sampled as observer
    FloatType sampling_prefactor;
    // factor for exponential moving average
    FloatType fading;
    // number of nearest observers to consider
    std::size_t neighbor_cnt;

    // counter of processed samples
    int last_index;
    // counter index when we sampled the last time
    int last_added_index;
    // time when we last sampled
    FloatType last_added_time;

    DistanceFunction distance_function;
    std::mt19937 rng;

    struct Observer {
        Vector<FloatType> data;
        FloatType observations;
        FloatType time_touched;
        FloatType time_added;
        int index;
    };
    
    struct ObserverCompare{
        FloatType fading;
        ObserverCompare(FloatType fading) : fading(fading) {}
        bool operator()(const Observer& a, const Observer& b) const {
            FloatType common_touched = std::max(a.time_touched, b.time_touched);
            
            FloatType observations_a = a.observations
                * std::pow(fading, common_touched - a.time_touched);
            
            FloatType observations_b = b.observations
                * std::pow(fading, common_touched - b.time_touched);
            
            // tie breaker for reproducibility
            if (observations_a == observations_b)
                return a.index < b.index;
            return observations_a > observations_b;
        }
    } observer_compare;
    
    struct ObserverAvCompare{
        FloatType fading;
        ObserverAvCompare(FloatType fading) : fading(fading) {}
        bool operator()(FloatType now, const Observer& a, const Observer& b) {
            FloatType common_touched = std::max(a.time_touched, b.time_touched);
            
            FloatType observations_a = a.observations * std::pow(fading, common_touched - a.time_touched);
            FloatType age_a = 1-std::pow(fading, now-a.time_added);
            
            FloatType observations_b = b.observations * std::pow(fading, common_touched - b.time_touched);
            FloatType age_b = 1-std::pow(fading, now-b.time_added);
            
            // do not necessarily need a tie breaker here
            return observations_a * age_b > observations_b * age_a;
        }
    } observer_av_compare;
    
    typedef boost::container::multiset<Observer,ObserverCompare> MapType;
    typedef typename MapType::iterator MapIterator;
    MapType observers;

    FloatType fitPredict_impl(const Vector<FloatType>& data, FloatType now, bool fit_only) {
        FloatType score = 0; // returned for first seen sample
        std::vector<std::pair<FloatType,MapIterator>> nearest;
        
        int i = 0;
        int active_threshold = (observers.size()-1) * active_observers;
        MapIterator worst_observer = observers.begin();
        
        auto cmp = [] (const std::pair<FloatType,MapIterator>& a, const std::pair<FloatType,MapIterator>& b) {
            // in case of ties prefer older observers
            if (a.first == b.first)
                return a.second->index < b.second->index;
            return a.first < b.first;
        };
        FloatType observations_sum = 0;
        for (auto it = observers.begin(); it != observers.end(); it++) {
            FloatType distance = distance_function(data, it->data);
            observations_sum += it->observations * std::pow<FloatType>(fading, now-it->time_touched);
            if (nearest.size() < neighbor_cnt) {
                nearest.emplace_back(distance, it);
                std::push_heap(nearest.begin(), nearest.end(), cmp);
            }
            else if (nearest[0].first > distance) {
                std::pop_heap(nearest.begin(), nearest.end(), cmp);
                nearest.back().first = distance;
                nearest.back().second = it;
                std::push_heap(nearest.begin(), nearest.end(), cmp);
            }
            if (i == active_threshold && !fit_only) {
                int len = nearest.size();
                std::vector<FloatType> sorted_nearest(len);
                for (int j = 0; j < len; j++)
                    sorted_nearest[j] = nearest[j].first;
                std::sort_heap(sorted_nearest.begin(), sorted_nearest.end());
                if (len % 2 == 0) {
                    score = (sorted_nearest[len/2] + sorted_nearest[len/2 + 1]) / 2;
                }
                else {
                    score = sorted_nearest[len/2];
                }
            }
            if (observer_av_compare(now, *worst_observer, *it)) {
                worst_observer = it;
            }
            i++;
        }
        FloatType observations_nearest_sum = 0;
        for (auto& observed : nearest) {
            MapIterator it = observed.second;
            auto node = observers.extract(it);
            Observer& observer = node.value();
            observer.observations *= std::pow<FloatType>(fading, now-observer.time_touched);
            observer.observations += 1;
            observer.time_touched = now;
            observations_nearest_sum += observer.observations;
            observers.insert(std::move(node));
            // take into account that observations have been
            // incremented by 1 for observations_nearest_sum
            observations_sum += 1;
        }
        // Note: observations_nearest_sum == NaN might happen
        bool add_as_observer = 
            observers.empty() ||
            (rng() - rng.min()) * observations_sum * (last_index - last_added_index) < sampling_prefactor * (rng.max() - rng.min()) * observations_nearest_sum * (now - last_added_time) ;

        if (add_as_observer) {
            if (observers.size() < observer_cnt) {
                observers.insert({data, 1, now, now, last_index});
            }
            else {
                auto node = observers.extract(worst_observer);
                Observer& observer = node.value();
                observer.data = data;
                observer.observations = 1;
                observer.time_touched = now;
                observer.time_added = now;
                observer.index = last_index;
                observers.insert(std::move(node));
            }
            last_added_index = last_index;
            last_added_time = now;
        }
        last_index++;
        return score;
    }
    
  public:
    SDOstream(std::size_t observer_cnt, FloatType T, FloatType idle_observers, std::size_t neighbor_cnt, DistanceFunction distance_function=Vector<FloatType>::euclidean, int seed=0) :
      observer_cnt(observer_cnt), 
      active_observers(1-idle_observers), 
      sampling_prefactor(observer_cnt * observer_cnt / neighbor_cnt / T),
      fading(std::exp(-1/T)),
      neighbor_cnt(neighbor_cnt),
      last_index(0),
      last_added_index(0),
      distance_function(distance_function),
      rng(seed),
      observer_compare(fading),
      observer_av_compare(fading),
      observers(observer_compare)
    { }
            
    void fit(const Vector<FloatType>& data, FloatType now) {
        fitPredict_impl(data, now, true);
    }

    FloatType fitPredict(const Vector<FloatType>& data, FloatType now) {
        return fitPredict_impl(data, now, false);
    }
    
    int observerCount() { return observers.size(); }
    
    bool lastWasSampled() { return last_added_index == last_index - 1; }
    
    class ObserverView{
        FloatType fading;
        MapIterator it;
    public:
        ObserverView(FloatType fading, MapIterator it) :
            fading(fading),
            it(it)
        { }
        Vector<FloatType> getData() { return it->data; }
        FloatType getObservations(FloatType now) {
            return it->observations * std::pow(fading, now - it->time_touched);
        }
        FloatType getAvObservations(FloatType now) {
            return (1-fading) * it->observations * std::pow(fading, now - it->time_touched) /
                (1-std::pow(fading, now - it->time_added));
        }
    };
    
    class iterator : public MapIterator {
        FloatType fading;
      public:
        ObserverView operator*() { return ObserverView(fading, MapIterator(*this)); };
        iterator() {}
        iterator(FloatType fading, MapIterator it) : 
            MapIterator(it),
            fading(fading)
        { }
    };
    
    iterator begin() { return iterator(fading, observers.begin()); }
    iterator end() { return iterator(fading, observers.end()); }

};

#endif

