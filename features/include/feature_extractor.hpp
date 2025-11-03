#pragma once

#include "types.hpp"
#include "hash_utils.hpp"
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <random>
#include <limits>

namespace step {

/**
 * FeatureExtractor implements Algorithm 2: Computation of STEP Feature Vectors
 *
 * For each event, computes feature vector based on motif extensions and their
 * normalized posterior probabilities.
 */
class FeatureExtractor {
public:
    explicit FeatureExtractor(int max_length, double delta_c = 0.0)
        : max_motif_length_(max_length), delta_c_(delta_c) {}

    /**
     * Extract features from chronological events
     * Returns:
     * - feature_matrix: Φ ∈ R^{|E|×M} where M is number of unique motif types
     * - motif_index: mapping from motif type string to column index
     * - events: all events (positive + negative if enabled)
     * - labels: 1 for positive (real), 0 for negative (sampled)
     */
    struct FeatureResult {
        FeatureMatrix feature_matrix;
        MotifIndex motif_to_index;
        std::vector<MotifType> index_to_motif;  // reverse mapping
        std::vector<Event> events;  // All events (positive + negative)
        std::vector<int> labels;    // 1 = positive, 0 = negative
        double delta_c;
    };

    FeatureResult extract_features(const std::vector<Event>& events,
                                   bool enable_negative_sampling = false,
                                   double negative_ratio = 1.0,
                                   int random_seed = 42) {
        FeatureResult result;

        // Calculate delta_c if not provided
        if (delta_c_ <= 0.0) {
            result.delta_c = calculate_delta_c(events);
        } else {
            result.delta_c = delta_c_;
        }

        // Prepare events and labels
        std::vector<Event> all_events = events;
        std::vector<int> labels(events.size(), 1);  // All positive initially

        // Generate negative samples if enabled
        if (enable_negative_sampling && negative_ratio > 0.0) {
            auto negative_samples = generate_negative_samples(events, negative_ratio, random_seed);
            all_events.insert(all_events.end(), negative_samples.begin(), negative_samples.end());
            labels.insert(labels.end(), negative_samples.size(), 0);  // 0 = negative
        }

        // First pass: collect statistics for probability computation (only from positive samples)
        MotifStats stats = collect_statistics(events, result.delta_c);

        // Build motif index
        int motif_idx = 0;
        for (const auto& [motif_type, _] : stats.motif_occurrences) {
            result.motif_to_index[motif_type] = motif_idx;
            result.index_to_motif.push_back(motif_type);
            motif_idx++;
        }

        // Initialize feature matrix: |E| × M (all zeros)
        const size_t num_events = all_events.size();
        const size_t num_motif_types = result.motif_to_index.size();
        result.feature_matrix.resize(num_events, FeatureVector(num_motif_types, 0.0));

        // Store events and labels in result
        result.events = all_events;
        result.labels = labels;

        // Second pass: compute feature vectors using Algorithm 2 for all events
        compute_feature_vectors(all_events, stats, result);

        return result;
    }

    // Utility function to get motif code (same as sequential)
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

private:
    int max_motif_length_;
    double delta_c_;

    /**
     * Generate negative samples using random sampling
     * For each positive event, generates neg_ratio negative samples with:
     * - Same timestamp as the corresponding positive event
     * - Different edges (non-occurring edges)
     */
    std::vector<Event> generate_negative_samples(const std::vector<Event>& positive_events,
                                                  double negative_ratio,
                                                  int random_seed) {
        std::mt19937 rng(random_seed);

        // Collect all unique nodes and create edge set from positive events
        std::unordered_set<Node> nodes;
        std::unordered_set<Edge> positive_edges;

        for (const auto& event : positive_events) {
            nodes.insert(event.first.first);
            nodes.insert(event.first.second);
            positive_edges.insert(event.first);
        }

        std::vector<Node> node_list(nodes.begin(), nodes.end());
        std::uniform_int_distribution<size_t> node_dist(0, node_list.size() - 1);

        std::vector<Event> negative_samples;
        const size_t neg_per_positive = static_cast<size_t>(negative_ratio);
        const double fractional_part = negative_ratio - neg_per_positive;

        // For fractional ratios, randomly decide if we generate one more sample
        std::uniform_real_distribution<double> frac_dist(0.0, 1.0);

        // Generate negative samples for each positive event
        for (const auto& positive_event : positive_events) {
            const Timestamp timestamp = positive_event.second;

            // Calculate how many negatives to generate for this positive event
            size_t num_neg_for_this = neg_per_positive;
            if (fractional_part > 0.0 && frac_dist(rng) < fractional_part) {
                num_neg_for_this++;
            }

            // Generate num_neg_for_this negative samples with the same timestamp
            size_t generated = 0;
            size_t attempts = 0;
            const size_t max_attempts = num_neg_for_this * 100;  // Avoid infinite loops

            while (generated < num_neg_for_this && attempts < max_attempts) {
                attempts++;

                // Sample two random nodes
                Node u = node_list[node_dist(rng)];
                Node v = node_list[node_dist(rng)];

                // Skip self-loops
                if (u == v) continue;

                // Create edge
                Edge edge = {u, v};

                // Check if this edge exists in positive samples
                if (positive_edges.find(edge) != positive_edges.end()) {
                    continue;  // Edge exists, try another
                }

                // Check reverse edge too
                Edge reverse_edge = {v, u};
                if (positive_edges.find(reverse_edge) != positive_edges.end()) {
                    continue;  // Reverse edge exists, try another
                }

                // Add negative sample with same timestamp as positive event
                negative_samples.push_back({edge, timestamp});
                generated++;
            }
        }

        // Sort by timestamp to maintain chronological order
        std::sort(negative_samples.begin(), negative_samples.end(),
                  [](const Event& a, const Event& b) {
                      return a.second < b.second;
                  });

        return negative_samples;
    }

    struct MotifStats {
        // Transition counts: old_motif_code -> new_motif_code -> count
        std::unordered_map<std::string, std::unordered_map<std::string, int>> transition_counts;

        // Inter-event times for each motif type
        std::unordered_map<std::string, std::vector<Timestamp>> inter_event_times;

        // All motif types that occurred
        std::unordered_map<std::string, int> motif_occurrences;
    };

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

    /**
     * Collect statistics from events for probability computation
     */
    MotifStats collect_statistics(const std::vector<Event>& events, double delta_c) {
        MotifStats stats;
        std::unordered_set<Motif> motifs_open;

        for (const auto& event : events) {
            process_event_for_stats(event, motifs_open, stats, delta_c);
        }

        return stats;
    }

    void process_event_for_stats(const Event& event,
                                  std::unordered_set<Motif>& motifs_open,
                                  MotifStats& stats,
                                  double delta_c) {
        Node event_u = event.first.first;
        Node event_v = event.first.second;
        Timestamp event_time = event.second;

        std::vector<Motif> motifs_closed;
        std::vector<Motif> motifs_opened;

        for (const auto& motif : motifs_open) {
            // Check if motif should be closed
            if (event_time - motif.back().second > delta_c ||
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

            // Valid extension
            Motif motif_new = motif;
            motif_new.push_back(event);

            std::string old_code = get_motif_code(motif).first;
            std::string new_code = get_motif_code(motif_new).first;

            stats.transition_counts[old_code][new_code]++;
            stats.inter_event_times[new_code].push_back(event_time - motif.back().second);
            stats.motif_occurrences[new_code]++;

            motifs_opened.push_back(motif_new);
            motifs_closed.push_back(motif);
        }

        // New single-event motif
        motifs_open.insert({event});
        std::string single_code = get_motif_code({event}).first;
        stats.motif_occurrences[single_code]++;

        // Update open motifs
        for (const auto& motif : motifs_closed) {
            motifs_open.erase(motif);
        }
        for (const auto& motif : motifs_opened) {
            motifs_open.insert(motif);
        }
    }

    /**
     * Algorithm 2: Compute STEP Feature Vectors
     */
    void compute_feature_vectors(const std::vector<Event>& events,
                                  const MotifStats& stats,
                                  FeatureResult& result) {
        std::unordered_set<Motif> open_motifs;  // O in the algorithm

        for (size_t i = 0; i < events.size(); ++i) {
            const Event& ei = events[i];
            const Timestamp Ti = ei.second;

            // Step 7: Update O by removing expired instances
            remove_expired_motifs(open_motifs, Ti, result.delta_c);

            // Step 8: Let C = {mold ∈ O : mold ∪ {ei} valid}
            std::vector<Motif> candidates = find_candidates(open_motifs, ei, Ti, result.delta_c);

            // Steps 9-13: For each candidate, compute posterior and update feature vector
            for (const auto& mold : candidates) {
                Motif mnew = mold;
                mnew.push_back(ei);

                // Get motif codes
                std::string old_code = get_motif_code(mold).first;
                std::string new_code = get_motif_code(mnew).first;

                // Compute normalized posterior via Eq. (3.5)
                double posterior = compute_normalized_posterior(mold, mnew, Ti, stats);

                // Assign Φ[i, index(mnew)] ← P(mold → mnew | Ti)
                if (result.motif_to_index.find(new_code) != result.motif_to_index.end()) {
                    int motif_idx = result.motif_to_index[new_code];
                    result.feature_matrix[i][motif_idx] = posterior;
                }
            }

            // Step 14: Insert new motif instances into O
            update_open_motifs(open_motifs, candidates, ei);
        }
    }

    void remove_expired_motifs(std::unordered_set<Motif>& open_motifs,
                                Timestamp current_time,
                                double delta_c) {
        std::vector<Motif> to_remove;
        for (const auto& motif : open_motifs) {
            if (current_time - motif.back().second > delta_c) {
                to_remove.push_back(motif);
            }
        }
        for (const auto& motif : to_remove) {
            open_motifs.erase(motif);
        }
    }

    std::vector<Motif> find_candidates(const std::unordered_set<Motif>& open_motifs,
                                        const Event& event,
                                        Timestamp event_time,
                                        double delta_c) {
        std::vector<Motif> candidates;
        Node event_u = event.first.first;
        Node event_v = event.first.second;

        for (const auto& motif : open_motifs) {
            // Check time constraint
            if (event_time - motif.back().second > delta_c) {
                continue;
            }

            // Check length constraint
            if (motif.size() >= static_cast<size_t>(max_motif_length_)) {
                continue;
            }

            // Skip if same edge
            if (event.first == motif.back().first) {
                continue;
            }

            // Check if event shares nodes with motif (valid extension)
            std::unordered_set<Node> motif_nodes;
            for (const auto& motif_event : motif) {
                motif_nodes.insert(motif_event.first.first);
                motif_nodes.insert(motif_event.first.second);
            }

            if (motif_nodes.find(event_u) != motif_nodes.end() ||
                motif_nodes.find(event_v) != motif_nodes.end()) {
                candidates.push_back(motif);
            }
        }

        return candidates;
    }

    /**
     * Compute normalized posterior P(mold → mnew | Ti)
     * Following typical STEP formulation:
     * P = P(transition) * P(time | transition)
     */
    double compute_normalized_posterior(const Motif& mold,
                                         const Motif& mnew,
                                         Timestamp Ti,
                                         const MotifStats& stats) {
        std::string old_code = get_motif_code(mold).first;
        std::string new_code = get_motif_code(mnew).first;

        // P(transition): count(old -> new) / sum of all transitions from old
        double p_transition = 0.0;
        auto trans_it = stats.transition_counts.find(old_code);
        if (trans_it != stats.transition_counts.end()) {
            int total_from_old = 0;
            for (const auto& [_, count] : trans_it->second) {
                total_from_old += count;
            }

            auto new_it = trans_it->second.find(new_code);
            if (new_it != trans_it->second.end() && total_from_old > 0) {
                p_transition = static_cast<double>(new_it->second) / total_from_old;
            }
        }

        // P(time | transition): exponential distribution
        double p_time = 0.0;
        auto iet_it = stats.inter_event_times.find(new_code);
        if (iet_it != stats.inter_event_times.end() && !iet_it->second.empty()) {
            // Calculate mean inter-event time
            double mean_iet = 0.0;
            for (Timestamp t : iet_it->second) {
                mean_iet += t;
            }
            mean_iet /= iet_it->second.size();

            // Exponential density: λ * exp(-λ * t)
            if (mean_iet > 0) {
                double lambda = 1.0 / mean_iet;
                Timestamp delta_t = Ti - mold.back().second;
                p_time = lambda * std::exp(-lambda * delta_t);
            }
        }

        // Combined probability (normalized posterior)
        return p_transition * p_time;
    }

    void update_open_motifs(std::unordered_set<Motif>& open_motifs,
                            const std::vector<Motif>& old_motifs,
                            const Event& event) {
        // Remove old motifs that were extended
        for (const auto& motif : old_motifs) {
            open_motifs.erase(motif);
        }

        // Add extended motifs
        for (const auto& motif : old_motifs) {
            Motif extended = motif;
            extended.push_back(event);
            open_motifs.insert(extended);
        }

        // Always add single-event motif
        open_motifs.insert({event});
    }
};

} // namespace step
