#pragma once

#include "types.hpp"
#include "hash_utils.hpp"
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <iostream>

namespace step {

struct EvaluationMetrics {
    int true_positives = 0;
    int false_positives = 0;
    int false_negatives = 0;
    double precision = 0.0;
    double recall = 0.0;
    double f1_score = 0.0;

    void calculate() {
        if (true_positives + false_positives > 0) {
            precision = static_cast<double>(true_positives) / (true_positives + false_positives);
        }
        if (true_positives + false_negatives > 0) {
            recall = static_cast<double>(true_positives) / (true_positives + false_negatives);
        }
        if (precision + recall > 0) {
            f1_score = 2.0 * (precision * recall) / (precision + recall);
        }
    }

    void print() const {
        std::cout << "Precision: " << precision << "\n";
        std::cout << "True Positives: " << true_positives << "\n";
        std::cout << "False Positives: " << false_positives << "\n";
        std::cout << "Recall: " << recall << "\n";
        std::cout << "False Negatives: " << false_negatives << "\n";
        std::cout << "F1 Score: " << f1_score << "\n";
    }

    std::string to_csv() const {
        return std::to_string(precision) + "," +
               std::to_string(true_positives) + "," +
               std::to_string(false_positives) + "," +
               std::to_string(recall) + "," +
               std::to_string(false_negatives) + "," +
               std::to_string(f1_score);
    }
};

class EventScorer {
public:
    struct DataSplit {
        std::unordered_set<Edge> train_edges;
        std::unordered_set<Edge> truth_edges;
        std::unordered_set<Node> train_nodes;
        std::unordered_set<Node> truth_nodes;
    };

    static DataSplit split_events(const std::vector<Event>& events, double train_ratio) {
        DataSplit split;

        for (size_t i = 0; i < events.size() * train_ratio; ++i) {
            const auto& [edge, _] = events[i];
            split.train_edges.insert(edge);
            split.train_nodes.insert(edge.first);
            split.train_nodes.insert(edge.second);
        }

        for (size_t i = static_cast<size_t>(events.size() * train_ratio); i < events.size(); ++i) {
            const auto& [edge, _] = events[i];
            split.truth_edges.insert(edge);
            split.truth_nodes.insert(edge.first);
            split.truth_nodes.insert(edge.second);
        }

        return split;
    }

    static EvaluationMetrics evaluate(
        const std::unordered_set<Edge>& truth_edges,
        const std::unordered_set<Edge>& predicted_edges
    ) {
        EvaluationMetrics metrics;
        std::unordered_set<Edge> processed_edges;

        // Calculate TP and FN from truth edges
        for (const auto& edge : truth_edges) {
            if (processed_edges.count(edge)) {
                continue;
            }
            processed_edges.insert(edge);

            if (predicted_edges.count(edge)) {
                metrics.true_positives++;
            } else {
                metrics.false_negatives++;
            }
        }

        // Calculate FP from predicted edges
        processed_edges.clear();
        for (const auto& edge : predicted_edges) {
            if (processed_edges.count(edge)) {
                continue;
            }
            processed_edges.insert(edge);

            if (!truth_edges.count(edge)) {
                metrics.false_positives++;
            }
        }

        metrics.calculate();
        return metrics;
    }
};

} // namespace step
