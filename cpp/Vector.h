// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

#ifndef DSALMON_VECTOR_H
#define DSALMON_VECTOR_H

#include "util.h"

#include <assert.h>
#include <cmath>
#include <vector>

#include <istream>
#include <ostream>

template<typename Value>
class Vector : public std::vector<Value>
{
  public:
    Vector() {}
    Vector(int dimension) : std::vector<Value>(dimension, 0) {}
    Vector(Value* data, int dimension) : std::vector<Value>(data, data + dimension) {}

    Vector(Vector&&) = default;
    Vector(const Vector&) = default;
    Vector& operator=(const Vector&) = default;
    
    Vector operator+(const Vector& other) const {
        Vector result(this->size());
        for (int i = 0; i < this->size(); i++)
            result[i] = (*this)[i] + other[i];
        return result;
    }
    
    Vector operator-(const Vector& other) const {
        Vector result(this->size());
        for (int i = 0; i < this->size(); i++)
            result[i] = (*this)[i] - other[i];
        return result;
    }
    
    Vector& operator+=(const Vector& other) {
        for (int i = 0; i < this->size(); i++)
            (*this)[i] += other[i];
        return *this;
    }
    
    Vector& operator-=(const Vector& other) {
        for (int i = 0; i < this->size(); i++)
            (*this)[i] -= other[i];
        return *this;
    }

    static Value chebyshev(const Vector& a, const Vector& b) {
        Value distance = 0;
        assert(a.size() == b.size());
        for (std::size_t i = 0; i < a.size(); i++) {
            Value diff = a[i] - b[i];
            distance = std::max(distance, std::abs(diff));
        }
        return distance;
    }
    
    static Value euclidean(const Vector& a, const Vector& b) {
        Value distance = 0;
        assert(a.size() == b.size());
        // TODO: here it makes sense to use a signed int to get maximum
        // performance. However, this is a bit inconsistent with the remaining code
        int i = 0;
        int len = a.size();
        for (; i < len; i++) {
            Value diff = a[i] - b[i];
            distance += diff * diff;
        }
        return std::sqrt(distance);
    }

    static Value manhattan(const Vector& a, const Vector& b) {
        Value distance = 0;
        assert(a.size() == b.size());
        for (std::size_t i = 0; i < a.size(); i++) {
            Value diff = a[i] - b[i];
            distance += std::abs(diff);
        }
        return distance;
    }

    static Value lp(const Vector& a, const Vector& b, const Value p) {
        Value distance = 0;
        assert(a.size() == b.size());
        for (std::size_t i = 0; i < a.size(); i++) {
            Value diff = a[i] - b[i];
            distance += std::pow(std::abs(diff), p);
        }
        return std::pow(distance, 1/p);
    }

    void unserialize(std::istream& in) {
        std::size_t len = unserializeInt<std::uint64_t>(in);
        this->resize(len);
        in.read(
            reinterpret_cast<char*>(this->data()),
            this->size() * sizeof(*this->data())
        );
    }

    void serialize(std::ostream& out) const {
        serializeInt<std::uint64_t>(out, this->size());
        out.write(
            reinterpret_cast<const char*>(this->data()),
            this->size() * sizeof(*this->data())
        );
    }
};

typedef Vector<double> doubleVector;

#endif
