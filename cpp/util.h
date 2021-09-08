// Copyright (c) 2020 CN Group, TU Wien
// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

#ifndef DSALMON_UTIL_H
#define DSALMON_UTIL_H

#include <iostream>

// Serialize integers as big endians
template<typename T>
void serializeInt(std::ostream& os, T data) {
    char buffer[sizeof(T)];
    for (std::size_t i = sizeof(T); i > 0; i--) {
        buffer[i-1] = data&0xff;
        data >>= 8;
    }
    os.write(buffer, sizeof(buffer));
}

template<typename T>
T unserializeInt(std::istream& is) {
    char buffer[sizeof(T)];
    T data = 0;
    is.read(buffer, sizeof(buffer));
    for (std::size_t i = 0; i < sizeof(T); i++) {
        data = (data<<8) | (unsigned char)buffer[i];
    }
    return data;
}

// Serialize floating point numbers in native system format
template<typename T>
void serializeDistance(std::ostream& os, T distance) {
    os.write(reinterpret_cast<char*>(&distance), sizeof(distance));
}

template<typename T>
T unserializeDistance(std::istream& is) {
    T distance;
    is.read(reinterpret_cast<char*>(&distance), sizeof(distance));
    return distance;
}

#endif
