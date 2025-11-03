#pragma once

#include "types.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <stdexcept>

namespace step {

class EventReader {
public:
    struct ReadResult {
        std::vector<Event> events;
        Timestamp base_time = 0;
    };

    static ReadResult read_events(const std::string& file_path) {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + file_path);
        }

        std::vector<Event> all_events;
        Node u, v;
        Timestamp tmx;

        while (file >> u >> v >> tmx) {
            if (u == v) {
                continue; // Skip self-loops
            }
            all_events.push_back({{u, v}, tmx});
        }

        if (all_events.empty()) {
            throw std::runtime_error("No valid events found in file: " + file_path);
        }

        // Sort by timestamp
        std::sort(all_events.begin(), all_events.end(),
                  [](const Event& a, const Event& b) {
                      return a.second < b.second;
                  });

        // Remove duplicates
        all_events.erase(std::unique(all_events.begin(), all_events.end()),
                        all_events.end());

        ReadResult result;
        result.base_time = all_events[0].second;

        // Normalize timestamps (subtract base time)
        for (auto& event : all_events) {
            event.second -= result.base_time;
        }

        result.events = std::move(all_events);
        return result;
    }
};

} // namespace step
