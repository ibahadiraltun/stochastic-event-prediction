#include "config.hpp"
#include "event_reader.hpp"
#include "feature_extractor.hpp"
#include "shared_memory.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <memory>

using namespace step;

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n"
              << "Options:\n"
              << "  -i <file>         Input file path (required)\n"
              << "  -o <file>         Output file path (required)\n"
              << "  -m <int>          Max motif length (default: 3)\n"
              << "  -d <double>       Delta C time window (default: auto-calculate)\n"
              << "  --negative        Enable negative sampling for classifier\n"
              << "  --neg-ratio <r>   Negative sampling ratio (default: 1.0)\n"
              << "  --seed <int>      Random seed for negative sampling (default: 42)\n"
              << "  --no-python       Disable Python predictor integration\n"
              << "  --shm-name <s>    Shared memory name (default: /step_embedding_shm)\n"
              << "  --timeout <ms>    Python timeout in milliseconds (default: 5000)\n"
              << "  -h, --help        Show this help message\n";
}

Config parse_args(int argc, char* argv[]) {
    Config config;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else if (arg == "-i" && i + 1 < argc) {
            config.input_file = argv[++i];
        } else if (arg == "-o" && i + 1 < argc) {
            config.output_file = argv[++i];
        } else if (arg == "-m" && i + 1 < argc) {
            config.max_motif_length = std::stoi(argv[++i]);
        } else if (arg == "-d" && i + 1 < argc) {
            config.delta_c = std::stod(argv[++i]);
        } else if (arg == "--no-python") {
            config.use_python_predictor = false;
        } else if (arg == "--shm-name" && i + 1 < argc) {
            config.shm_name = argv[++i];
        } else if (arg == "--timeout" && i + 1 < argc) {
            config.python_timeout_ms = std::stoi(argv[++i]);
        } else if (arg == "--negative") {
            config.enable_negative_sampling = true;
        } else if (arg == "--neg-ratio" && i + 1 < argc) {
            config.negative_sampling_ratio = std::stod(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            config.random_seed = std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            exit(1);
        }
    }

    return config;
}

void write_feature_matrix(const std::string& output_file,
                          const FeatureExtractor::FeatureResult& result) {
    std::ofstream out(output_file);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open output file: " + output_file);
    }

    // Write header with label and motif types
    out << "event_index,node_u,node_v,timestamp,label";
    for (const auto& motif_type : result.index_to_motif) {
        // Don't add "motif_" prefix to "gnn_output"
        if (motif_type == "gnn_output") {
            out << "," << motif_type;
        } else {
            out << ",motif_" << motif_type;
        }
    }
    out << "\n";

    // Write feature vectors for each event
    for (size_t i = 0; i < result.feature_matrix.size(); ++i) {
        const auto& event = result.events[i];
        out << i << ","
            << event.first.first << ","
            << event.first.second << ","
            << event.second << ","
            << result.labels[i];

        for (double feature : result.feature_matrix[i]) {
            out << "," << std::fixed << std::setprecision(6) << feature;
        }
        out << "\n";
    }

    std::cout << "Feature matrix written to: " << output_file << "\n";
}

int main(int argc, char* argv[]) {
    try {
        // Parse configuration
        Config config = parse_args(argc, argv);
        config.validate();

        std::cout << "STEP Embedding Feature Extractor\n";
        std::cout << "=================================\n";
        std::cout << "Input file: " << config.input_file << "\n";
        std::cout << "Output file: " << config.output_file << "\n";
        std::cout << "Max motif length: " << config.max_motif_length << "\n";
        std::cout << "Delta C: " << (config.delta_c > 0 ? std::to_string(config.delta_c) : "auto") << "\n";
        std::cout << "Negative sampling: " << (config.enable_negative_sampling ? "enabled" : "disabled");
        if (config.enable_negative_sampling) {
            std::cout << " (ratio: " << config.negative_sampling_ratio
                      << ", seed: " << config.random_seed << ")";
        }
        std::cout << "\n";
        std::cout << "Python predictor: " << (config.use_python_predictor ? "enabled" : "disabled") << "\n";
        std::cout << "\n";

        // Initialize shared memory if using Python predictor
        std::unique_ptr<SharedMemoryChannel> shm_channel;
        if (config.use_python_predictor) {
            std::cout << "Initializing shared memory: " << config.shm_name << "\n";
            shm_channel = std::make_unique<SharedMemoryChannel>(config.shm_name, config.shm_size);
            shm_channel->create();
            std::cout << "Shared memory created. Waiting for Python predictor to connect...\n";
            std::cout << "(Start your Python script now if not already running)\n\n";
        }

        // Read events
        std::cout << "Reading events from file...\n";
        auto read_result = EventReader::read_events(config.input_file);
        std::cout << "Loaded " << read_result.events.size() << " events\n";
        std::cout << "Base timestamp: " << read_result.base_time << "\n\n";

        // Extract features using Algorithm 2
        std::cout << "Extracting features using STEP Algorithm 2...\n";
        FeatureExtractor extractor(config.max_motif_length, config.delta_c);
        auto feature_result = extractor.extract_features(
            read_result.events,
            config.enable_negative_sampling,
            config.negative_sampling_ratio,
            config.random_seed
        );

        std::cout << "Feature extraction complete!\n";
        std::cout << "Total events: " << feature_result.events.size() << " ";
        std::cout << "(positive: " << std::count(feature_result.labels.begin(), feature_result.labels.end(), 1);
        std::cout << ", negative: " << std::count(feature_result.labels.begin(), feature_result.labels.end(), 0) << ")\n";
        std::cout << "Number of unique motif types: " << feature_result.motif_to_index.size() << "\n";
        std::cout << "Computed delta_c: " << feature_result.delta_c << "\n\n";

        // Optionally query Python for additional probabilities
        if (config.use_python_predictor && shm_channel) {
            std::cout << "Querying Python predictor for event probabilities...\n";
            int successful_requests = 0;
            int failed_requests = 0;

            // For each event, query Python neural network for independent probability
            for (size_t i = 0; i < feature_result.events.size(); ++i) {
                const auto& event = feature_result.events[i];

                try {
                    // Prepare request: raw event information "node_u node_v timestamp" (unnormalized)
                    Timestamp unnormalized_timestamp = event.second + read_result.base_time;
                    std::string event_info = std::to_string(event.first.first) + " " +
                                           std::to_string(event.first.second) + " " +
                                           std::to_string(unnormalized_timestamp);

                    // Query Python neural network
                    double python_prob = shm_channel->request_prediction(event_info, config.python_timeout_ms);

                    // Add Python probability as a new feature dimension
                    feature_result.feature_matrix[i].push_back(python_prob);

                    successful_requests++;
                } catch (const std::exception& e) {
                    failed_requests++;
                    if (failed_requests == 1) {
                        std::cerr << "Warning: Python predictor error: " << e.what() << "\n";
                        std::cerr << "Continuing without Python augmentation...\n";
                        break;  // Stop trying after first failure
                    }
                }

                if (failed_requests > 0) {
                    break;  // Stop processing if Python failed
                }
            }

            // If we successfully got Python predictions, add column name
            if (successful_requests > 0 && failed_requests == 0) {
                feature_result.index_to_motif.push_back("gnn_output");
                feature_result.motif_to_index["gnn_output"] = feature_result.index_to_motif.size() - 1;
            }

            std::cout << "Python queries: " << successful_requests << " successful, "
                      << failed_requests << " failed\n\n";
        }

        // Write feature matrix to file
        std::cout << "Writing feature matrix to file...\n";
        write_feature_matrix(config.output_file, feature_result);

        // Cleanup
        if (shm_channel) {
            shm_channel->cleanup();
        }

        std::cout << "\nDone!\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
