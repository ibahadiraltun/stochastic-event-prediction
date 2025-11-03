#pragma once

#include "types.hpp"
#include "hash_utils.hpp"
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <vector>
#include <numeric>
#include <cassert>

namespace step {

class MotifProcessor {
public:
    struct MotifStats {
        std::unordered_map<std::string, std::unordered_map<std::string, int>> transition_counts;
        std::unordered_map<std::string, std::vector<Timestamp>> inter_event_times;
        std::vector<Event> cold_events;
        double delta_c = 0.0;
    };

    explicit MotifProcessor(int max_length) : max_motif_length_(max_length) {}

    MotifStats process_events(const std::vector<Event>& events, Timestamp global_time) {
        MotifStats stats;
        std::unordered_set<Motif> motifs_open;

        stats.delta_c = calculate_delta_c(events);

        for (const auto& event : events) {
            process_single_event(event, motifs_open, stats, global_time);
        }

        return stats;
    }

    static std::pair<std::string, std::unordered_map<Node, std::string>>
    get_motif_code(const Motif& motif) {
        std::string motif_code;
        std::unordered_map<Node, std::string> code;
        int idx = 0;

        for (const auto& event : motif) {
            Node u = event.first.first;
            Node v = event.first.second;

            if (code.find(u) == code.end()) {
                code[u] = std::to_string(idx++);
            }
            motif_code.append(code[u]);

            if (code.find(v) == code.end()) {
                code[v] = std::to_string(idx++);
            }
            motif_code.append(code[v]);
        }

        return {motif_code, code};
    }

    static double calculate_mean_iet(const std::vector<Timestamp>& timestamps,
                                     Timestamp global_time) {
        assert(!timestamps.empty());

        if (timestamps.size() == 1) {
            return static_cast<double>(global_time - timestamps[0]);
        }

        double iet_sum = 0.0;
        for (size_t i = 1; i < timestamps.size(); ++i) {
            iet_sum += timestamps[i] - timestamps[i - 1];
        }

        return iet_sum / (timestamps.size() - 1);
    }

private:
    int max_motif_length_;

    double calculate_delta_c(const std::vector<Event>& events) const {
        double delta = 0.0;

        for (size_t i = 1; i < events.size(); ++i) {
            const auto& curr = events[i].first;
            const auto& prev = events[i - 1].first;

            // Check if events share at least one node
            bool shares_node = (curr.first == prev.first ||
                              curr.first == prev.second ||
                              curr.second == prev.first ||
                              curr.second == prev.second);

            if (shares_node) {
                double time_diff = static_cast<double>(events[i].second - events[i - 1].second);
                delta = std::max(delta, time_diff);
            }
        }

        return delta;
    }

    void process_single_event(const Event& event,
                              std::unordered_set<Motif>& motifs_open,
                              MotifStats& stats,
                              Timestamp global_time) {
        Node event_u = event.first.first;
        Node event_v = event.first.second;
        Timestamp event_time = event.second;

        std::vector<Motif> motifs_closed;
        std::vector<Motif> motifs_opened;
        bool is_cold_event = true;

        for (const auto& motif : motifs_open) {
            // Check if motif should be closed
            if (event_time - motif.back().second > stats.delta_c ||
                motif.size() == static_cast<size_t>(max_motif_length_)) {
                motifs_closed.push_back(motif);
                continue;
            }

            // Skip if same edge
            if (event.first == motif.back().first) {
                continue;
            }

            // Check if event shares nodes with motif
            std::unordered_set<Node> motif_nodes;
            for (const auto& motif_event : motif) {
                motif_nodes.insert(motif_event.first.first);
                motif_nodes.insert(motif_event.first.second);
            }

            if (motif_nodes.find(event_u) == motif_nodes.end() &&
                motif_nodes.find(event_v) == motif_nodes.end()) {
                continue;
            }

            is_cold_event = false;
            Motif motif_new = motif;
            motif_new.push_back(event);

            std::string old_code = get_motif_code(motif).first;
            std::string new_code = get_motif_code(motif_new).first;

            stats.transition_counts[old_code][new_code]++;
            stats.inter_event_times[new_code].push_back(event_time - motif.back().second);

            motifs_opened.push_back(motif_new);
            motifs_closed.push_back(motif);
        }

        if (is_cold_event) {
            motifs_open.insert({event});
            stats.cold_events.push_back(event);
        }

        for (const auto& motif : motifs_closed) {
            motifs_open.erase(motif);
        }
        for (const auto& motif : motifs_opened) {
            motifs_open.insert(motif);
        }
    }
};

} // namespace step
