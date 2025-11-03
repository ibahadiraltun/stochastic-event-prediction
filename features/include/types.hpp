#pragma once

#include <utility>
#include <vector>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace step {

using Node = int;
using Timestamp = int64_t;
using Edge = std::pair<int, int>;
using Event = std::pair<Edge, Timestamp>;
using Motif = std::vector<Event>;

// Feature vector related types
using FeatureVector = std::vector<double>;
using FeatureMatrix = std::vector<FeatureVector>;

// Motif type identifier (e.g., "01", "01_12", etc.)
using MotifType = std::string;

// Index mapping for motif types
using MotifIndex = std::unordered_map<MotifType, int>;

} // namespace step
