// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

#ifndef DSALMON_VECTOR_H
#define DSALMON_VECTOR_H

#include "util.h"

#include <assert.h>
#include <cmath>
#include <cstdint>
#include <vector>

#include <istream>
#include <ostream>

#if defined(__SSE__) || defined(__AVX2__)
#include <immintrin.h>
#endif

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

#ifdef __SSE3__
    static Value euclideanSSE3(const Vector& a, const Vector& b);
#endif
#ifdef __AVX2__
    static Value euclideanAVX2(const Vector& a, const Vector& b);
#endif

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

#ifdef __SSE3__
template<>
inline float Vector<float>::euclideanSSE3(const Vector<float>& a, const Vector<float>& b) {
    int len = a.size();
    int remainder = len % 4;
    int i;

    __m128 sum = _mm_setzero_ps();

    // Process vectors in SIMD-sized chunks
    for (i = 0; i < len - remainder; i += 4) {
        __m128 vec1 = _mm_loadu_ps(&a[i]);
        __m128 vec2 = _mm_loadu_ps(&b[i]);

        __m128 diff = _mm_sub_ps(vec1, vec2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }

    // Process the remainder elements
    for (; i < len; ++i) {
        float diff = a[i] - b[i];
        sum[0] += diff * diff;
    }

    // Horizontal sum of the SIMD accumulator
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);

    // Extract the result and take the square root
    float result;
    _mm_store_ss(&result, sum);

    return std::sqrt(result);
}

template<>
inline double Vector<double>::euclideanSSE3(const Vector<double>& a, const Vector<double>& b) {
    int len = a.size();
    int remainder = len % 2;
    int i;

    __m128d sum = _mm_setzero_pd();

    // Process vectors in SSE-sized chunks
    for (i = 0; i < len - remainder; i += 2) {
        __m128d vec1 = _mm_loadu_pd(&a[i]);
        __m128d vec2 = _mm_loadu_pd(&b[i]);

        __m128d diff = _mm_sub_pd(vec1, vec2);
        sum = _mm_add_pd(sum, _mm_mul_pd(diff, diff));
    }

    // Process the remainder elements
    for (; i < len; ++i) {
        double diff = a[i] - b[i];
        sum[0] += diff * diff;
    }

    // Horizontal sum of the SSE accumulator
    sum = _mm_hadd_pd(sum, sum);

    // Extract the result and take the square root
    double result;
    _mm_store_sd(&result, sum);

    return std::sqrt(result);
}
#endif


#ifdef __AVX2__
template<>
inline float Vector<float>::euclideanAVX2(const Vector<float>& a, const Vector<float>& b) {
    int len = a.size();
    int remainder = len % 8;
    int i;

    __m256 sum = _mm256_setzero_ps();

    // Process vectors in AVX2-sized chunks
    for (i = 0; i < len - remainder; i += 8) {
        __m256 vec1 = _mm256_loadu_ps(&a[i]);
        __m256 vec2 = _mm256_loadu_ps(&b[i]);

        __m256 diff = _mm256_sub_ps(vec1, vec2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }

    // Process the remainder elements
    for (; i < len; ++i) {
        float diff = a[i] - b[i];
        sum[0] += diff * diff;
    }

    // Horizontal sum of the AVX2 accumulator
    sum = _mm256_hadd_ps(sum, sum);
    sum = _mm256_hadd_ps(sum, sum);

    // Extract the result and take the square root
    float result;
    _mm_store_ss(&result, _mm256_extractf128_ps(sum, 0));

    return std::sqrt(result);
}

template<>
inline double Vector<double>::euclideanAVX2(const Vector<double>& a, const Vector<double>& b) {
    int len = a.size();
    int remainder = len % 4;
    int i;

    __m256d sum = _mm256_setzero_pd();

    // Process vectors in AVX2-sized chunks
    for (i = 0; i < len - remainder; i += 4) {
        __m256d vec1 = _mm256_loadu_pd(&a[i]);
        __m256d vec2 = _mm256_loadu_pd(&b[i]);

        __m256d diff = _mm256_sub_pd(vec1, vec2);
        sum = _mm256_add_pd(sum, _mm256_mul_pd(diff, diff));
    }

    // Process the remainder elements
    for (; i < len; ++i) {
        double diff = a[i] - b[i];
        sum[0] += diff * diff;
    }

    // Horizontal sum of the AVX2 accumulator
    sum = _mm256_hadd_pd(sum, sum);
    sum = _mm256_hadd_pd(sum, sum);

    // Extract the lower 128 bits and store the scalar value
    __m128d result128 = _mm256_extractf128_pd(sum, 0);
    double result;
    _mm_store_sd(&result, result128);

    return std::sqrt(result);
}
#endif

typedef Vector<double> doubleVector;

#endif
