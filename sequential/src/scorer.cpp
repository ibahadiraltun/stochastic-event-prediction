#include "scorer.hpp"
#include "event_reader.hpp"
#include "hash_utils.hpp"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <stdexcept>

using namespace step;

void read_all_events(const std::string& file_path,
                     std::vector<Event>& events,
                     std::unordered_set<Edge>& edges,
                     std::unordered_set<Node>& nodes,
                     bool preserve_order = false) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + file_path);
    }

    Node u, v;
    int64_t tmx;

    while (file >> u >> v >> tmx) {
        if (u == v) {
            continue; // Skip self-loops
        }
        events.push_back({{u, v}, tmx});
        edges.insert({u, v});
        nodes.insert(u);
        nodes.insert(v);
    }

    if (preserve_order) {
        return;
    }

    // Sort by timestamp
    std::sort(events.begin(), events.end(),
              [](const Event& a, const Event& b) {
                  return a.second < b.second;
              });

    // Remove duplicates
    events.erase(std::unique(events.begin(), events.end()), events.end());
}

int main(int argc, char* argv[]) {
    try {
        if (argc < 5) {
            std::cerr << "Usage: " << argv[0]
                      << " <full_events_file> <predictions_file> <train_ratio> <test_ratio>\n";
            return 1;
        }

        std::string full_events_path = argv[1];
        std::string predictions_path = argv[2];
        double train_ratio = std::stod(argv[3]);
        double test_ratio = std::stod(argv[4]);

        // Validate ratios
        if (train_ratio < 0.0 || train_ratio > 1.0) {
            throw std::invalid_argument("Train ratio must be between 0 and 1 (inclusive)");
        }
        if (test_ratio < 0.0 || test_ratio > 1.0) {
            throw std::invalid_argument("Test ratio must be between 0 and 1 (inclusive)");
        }

        std::cout << "STEP Event Prediction Scorer\n";
        std::cout << "============================\n";
        std::cout << "Full events file: " << full_events_path << "\n";
        std::cout << "Predictions file: " << predictions_path << "\n";
        std::cout << "Train ratio: " << train_ratio << "\n";
        std::cout << "Test ratio: " << test_ratio << "\n\n";

        // Read full events (ground truth)
        std::vector<Event> full_events;
        std::unordered_set<Edge> full_edges;
        std::unordered_set<Node> full_nodes;
        read_all_events(full_events_path, full_events, full_edges, full_nodes, false);

        std::cout << "Total events in dataset: " << full_events.size() << "\n";

        // Read predictions (preserve original order)
        std::vector<Event> pred_events;
        std::unordered_set<Edge> pred_edges;
        std::unordered_set<Node> pred_nodes;
        read_all_events(predictions_path, pred_events, pred_edges, pred_nodes, true);

        std::cout << "Predicted events: " << pred_events.size() << "\n\n";

        // Split full events into train and test
        auto split = EventScorer::split_events(full_events, train_ratio);

        std::cout << "Training edges: " << split.train_edges.size() << "\n";
        std::cout << "Test edges (ground truth): " << split.truth_edges.size() << "\n";
        std::cout << "Predicted unique edges: " << pred_edges.size() << "\n\n";

        // Evaluate predictions
        auto metrics = EventScorer::evaluate(split.truth_edges, pred_edges);

        std::cout << "Evaluation Results\n";
        std::cout << "==================\n";
        metrics.print();

        std::cout << "\nCSV Format:\n";
        std::cout << metrics.to_csv() << "\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
