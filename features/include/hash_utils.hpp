#pragma once

#include <functional>
#include <utility>
#include <vector>

namespace step {

template <class T>
inline void hash_combine(std::size_t& seed, const T& v) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

} // namespace step

namespace std {
    template<typename S, typename T>
    struct hash<pair<S, T>> {
        inline size_t operator()(const pair<S, T>& v) const {
            size_t seed = 0;
            step::hash_combine(seed, v.first);
            step::hash_combine(seed, v.second);
            return seed;
        }
    };

    template<typename T>
    struct hash<vector<T>> {
        inline size_t operator()(const vector<T>& k) const {
            size_t seed = 0;
            for (const auto& elem : k) {
                step::hash_combine(seed, elem);
            }
            return seed;
        }
    };

    template<typename S, typename T>
    struct hash<vector<pair<S, T>>> {
        inline size_t operator()(const vector<pair<S, T>>& k) const {
            size_t seed = 0;
            for (const auto& elem : k) {
                step::hash_combine(seed, elem.first);
                step::hash_combine(seed, elem.second);
            }
            return seed;
        }
    };
}
