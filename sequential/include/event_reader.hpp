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
        std::vector<Event> train_events;
        std::vector<Timestamp> test_times;
        Timestamp base_time = 0;
    };

    static ReadResult read_events(const std::string& file_path, double train_ratio) {
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
        const Timestamp base_time = all_events[0].second;
        const size_t train_size = static_cast<size_t>(all_events.size() * train_ratio);

        // Store base time for potential denormalization
        result.base_time = base_time;

        // Create training events with normalized timestamps
        for (size_t i = 0; i < train_size; ++i) {
            result.train_events.push_back({
                all_events[i].first,
                all_events[i].second - base_time
            });
        }

        // Store test times
        for (size_t i = train_size; i < all_events.size(); ++i) {
            result.test_times.push_back(all_events[i].second - base_time);
        }

        return result;
    }
};

} // namespace step
