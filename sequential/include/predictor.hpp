#pragma once

#include "types.hpp"
#include "motif_processor.hpp"
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cmath>
#include <limits>
#include <iostream>

namespace step {

class Predictor {
public:
    explicit Predictor(unsigned seed = std::chrono::system_clock::now().time_since_epoch().count())
        : rng_(seed) {}

    std::vector<Event> predict(
        const MotifProcessor::MotifStats& stats,
        Timestamp global_time,
        double mean_iet_global,
        int prediction_size,
        int max_motif_length,
        const std::vector<Event>& all_events
    ) {
        std::vector<Event> predictions;
        std::unordered_set<Motif> motifs_open;

        // Calculate mean IETs for motifs
        std::unordered_map<std::string, double> mean_iet_motif;
        for (const auto& [motif_code, iets] : stats.inter_event_times) {
            mean_iet_motif[motif_code] = std::accumulate(iets.begin(), iets.end(), 0.0) / iets.size();
        }

        // Setup cold event distributions
        auto cold_edge_stats = setup_cold_edge_stats(all_events, global_time);

        std::exponential_distribution<> global_timer(1.0 / mean_iet_global);
        double cold_event_prob = static_cast<double>(stats.cold_events.size()) / all_events.size();
        std::bernoulli_distribution cold_pick_dist(cold_event_prob);

        Timestamp current_time = global_time;

        for (int i = 0; i < prediction_size; ++i) {
            current_time += static_cast<Timestamp>(global_timer(rng_));

            Edge selected_edge = {-1, -1};
            int retry_count = 0;
            const int max_retries = 1000;

            while (selected_edge.first == -1 && retry_count < max_retries) {
                bool is_cold_event = cold_pick_dist(rng_);

                if (is_cold_event) {
                    selected_edge = predict_cold_event(cold_edge_stats, current_time, motifs_open);
                } else {
                    selected_edge = predict_hot_event(
                        motifs_open,
                        stats,
                        mean_iet_motif,
                        current_time,
                        stats.delta_c,
                        max_motif_length
                    );
                }
                retry_count++;
            }

            // If we couldn't find an edge after max retries, stop predictions
            if (selected_edge.first == -1) {
                std::cerr << "Warning: Could not find valid edge after " << max_retries
                          << " attempts. Stopping at " << predictions.size() << " predictions.\n";
                break;
            }

            predictions.push_back({selected_edge, current_time});
        }

        return predictions;
    }

private:
    std::default_random_engine rng_;

    struct ColdEdgeStats {
        std::unordered_set<Edge> edges;
        std::unordered_map<Edge, int> frequency;
        std::unordered_map<Edge, std::vector<Timestamp>> times;
        std::unordered_map<Edge, double> mean_iet;
        int total_frequency = 0;
    };

    static double exponential_density_sum(double x, double lambda, double epsilon) {
        return std::exp(-lambda * (x - epsilon)) - std::exp(-lambda * (x + epsilon));
    }

    ColdEdgeStats setup_cold_edge_stats(const std::vector<Event>& events, Timestamp global_time) {
        ColdEdgeStats stats;

        for (const auto& event : events) {
            Edge edge = event.first;
            Edge edge_reverse = {edge.second, edge.first};
            Timestamp tmx = event.second;

            stats.times[edge].push_back(tmx);
            stats.times[edge_reverse].push_back(tmx);
            stats.frequency[edge]++;
            stats.total_frequency++;
            stats.edges.insert(edge);
        }

        for (const auto& [edge, times] : stats.times) {
            stats.mean_iet[edge] = MotifProcessor::calculate_mean_iet(times, global_time);
        }

        return stats;
    }

    Edge predict_cold_event(ColdEdgeStats& cold_stats,
                            Timestamp current_time,
                            std::unordered_set<Motif>& motifs_open) {
        double max_prob = std::numeric_limits<double>::lowest();
        Edge selected_edge = {-1, -1};

        for (const auto& edge : cold_stats.edges) {
            double p_edge = static_cast<double>(cold_stats.frequency[edge]) / cold_stats.total_frequency;
            double p_time = exponential_density_sum(
                current_time - cold_stats.times[edge].back(),
                1.0 / cold_stats.mean_iet[edge],
                1.0
            );
            double p_total = std::log(p_edge) + std::log(p_time);

            if (p_total > max_prob) {
                max_prob = p_total;
                selected_edge = edge;
            }
        }

        if (selected_edge.first != -1) {
            cold_stats.edges.erase(selected_edge);
            motifs_open.insert({{selected_edge, current_time}});
        }

        return selected_edge;
    }

    Edge predict_hot_event(
        std::unordered_set<Motif>& motifs_open,
        const MotifProcessor::MotifStats& stats,
        const std::unordered_map<std::string, double>& mean_iet_motif,
        Timestamp current_time,
        double delta_c,
        int max_motif_length
    ) {
        double max_prob = std::numeric_limits<double>::lowest();
        Edge selected_edge = {-1, -1};
        Motif selected_motif;

        std::vector<Motif> motifs_closed;

        for (const auto& base_motif : motifs_open) {
            if (current_time - base_motif.back().second > delta_c ||
                base_motif.size() == static_cast<size_t>(max_motif_length)) {
                motifs_closed.push_back(base_motif);
                continue;
            }

            auto [base_code, node_ids] = MotifProcessor::get_motif_code(base_motif);

            // Create reverse mapping
            std::unordered_map<std::string, Node> id_map;
            for (const auto& [node, id] : node_ids) {
                id_map[id] = node;
            }

            auto it = stats.transition_counts.find(base_code);
            if (it == stats.transition_counts.end()) {
                continue;
            }

            int64_t total_transitions = 0;
            for (const auto& [_, count] : it->second) {
                total_transitions += count;
            }

            for (const auto& [candidate_code, count] : it->second) {
                if (candidate_code.size() < 2) continue;

                std::string node_id_u = std::to_string(candidate_code[candidate_code.size() - 2] - '0');
                std::string node_id_v = std::to_string(candidate_code[candidate_code.size() - 1] - '0');

                if (id_map.find(node_id_u) == id_map.end() ||
                    id_map.find(node_id_v) == id_map.end()) {
                    continue;
                }

                Node u = id_map[node_id_u];
                Node v = id_map[node_id_v];

                double p_transition = static_cast<double>(count) / total_transitions;

                auto iet_it = mean_iet_motif.find(candidate_code);
                if (iet_it == mean_iet_motif.end()) {
                    continue;
                }

                double p_time = exponential_density_sum(
                    current_time - base_motif.back().second,
                    1.0 / iet_it->second,
                    1.0
                );

                double p_total = std::log(p_transition) + std::log(p_time);

                if (p_total > max_prob) {
                    max_prob = p_total;
                    selected_edge = {u, v};
                    selected_motif = base_motif;
                }
            }
        }

        // Clean up closed motifs
        for (const auto& motif : motifs_closed) {
            motifs_open.erase(motif);
        }

        if (selected_edge.first != -1) {
            motifs_open.erase(selected_motif);
            selected_motif.push_back({selected_edge, current_time});
            motifs_open.insert(selected_motif);
        }

        return selected_edge;
    }
};

} // namespace step
