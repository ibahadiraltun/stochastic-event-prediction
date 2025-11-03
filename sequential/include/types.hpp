#pragma once

#include <utility>
#include <vector>
#include <cstdint>

namespace step {

using Node = int;
using Timestamp = int64_t;
using Edge = std::pair<int, int>;
using Event = std::pair<Edge, Timestamp>;
using Motif = std::vector<Event>;

} // namespace step
