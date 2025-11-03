#include "config.hpp"
#include "event_reader.hpp"
#include "motif_processor.hpp"
#include "predictor.hpp"
#include "types.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>

using namespace step;

int main(int argc, char* argv[]) {
    try {
        // Parse command line arguments
        if (argc < 5) {
            std::cerr << "Usage: " << argv[0]
                      << " <input_file> <output_file> <train_ratio> <prediction_size>\n";
            return 1;
        }

        Config config;
        config.input_file = argv[1];
        config.output_file = argv[2];
        config.train_ratio = std::stod(argv[3]);
        config.prediction_size = std::stoi(argv[4]);
        config.validate();

        std::cout << "STEP Sequential Event Predictor\n";
        std::cout << "================================\n";
        std::cout << "Input file: " << config.input_file << "\n";
        std::cout << "Train ratio: " << config.train_ratio << "\n";
        std::cout << "Prediction size: " << config.prediction_size << "\n\n";

        // Read events
        auto read_result = EventReader::read_events(config.input_file, config.train_ratio);
        auto& all_events_train = read_result.train_events;

        if (all_events_train.empty()) {
            throw std::runtime_error("No training events available");
        }

        Timestamp global_time = all_events_train.back().second;

        // Filter events based on use_ratio
        std::vector<Event> events;
        size_t start_idx = static_cast<size_t>(all_events_train.size() * config.use_ratio);
        for (size_t i = start_idx; i < all_events_train.size(); ++i) {
            events.push_back(all_events_train[i]);
        }

        std::cout << "Total training events: " << all_events_train.size() << "\n";
        std::cout << "Events used for motif learning: " << events.size() << "\n";
        std::cout << "Last training event timestamp (normalized): " << all_events_train.back().second << "\n";
        std::cout << "Last training event timestamp (original): " << (all_events_train.back().second + read_result.base_time) << "\n";
        if (!read_result.test_times.empty()) {
            std::cout << "First test event timestamp (normalized): " << read_result.test_times.front() << "\n";
            std::cout << "First test event timestamp (original): " << (read_result.test_times.front() + read_result.base_time) << "\n";
            std::cout << "Last test event timestamp (normalized): " << read_result.test_times.back() << "\n";
            std::cout << "Last test event timestamp (original): " << (read_result.test_times.back() + read_result.base_time) << "\n";
        }

        // Extract event times for global mean IET calculation
        std::vector<Timestamp> event_times;
        event_times.reserve(events.size());
        for (const auto& event : events) {
            event_times.push_back(event.second);
        }

        double mean_iet_global = MotifProcessor::calculate_mean_iet(event_times, global_time);
        std::cout << "Global mean inter-event time: " << mean_iet_global << "\n\n";

        // Process motifs
        std::cout << "Processing motifs...\n";
        auto start_time = std::chrono::system_clock::now();

        MotifProcessor processor(config.max_motif_length);
        auto stats = processor.process_events(events, global_time);

        auto end_time = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "Motif initialization time: " << duration.count() << " ms\n";
        std::cout << "Cold events discovered: " << stats.cold_events.size() << "\n";
        std::cout << "Unique motif codes: " << stats.inter_event_times.size() << "\n";
        std::cout << "Delta C (connectivity window): " << stats.delta_c << "\n\n";

        // Make predictions
        std::cout << "Generating predictions...\n";
        start_time = std::chrono::system_clock::now();

        Predictor predictor;
        auto predictions = predictor.predict(
            stats,
            global_time,
            mean_iet_global,
            config.prediction_size,
            config.max_motif_length,
            events
        );

        end_time = std::chrono::system_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "Prediction time: " << duration.count() << " ms\n";

        // Write predictions to output file
        std::ofstream output(config.output_file);
        if (!output.is_open()) {
            throw std::runtime_error("Failed to open output file: " + config.output_file);
        }

        // Denormalize timestamps if configured
        Timestamp time_offset = config.denormalize_timestamps ? read_result.base_time : 0;

        for (const auto& [edge, timestamp] : predictions) {
            output << edge.first << " " << edge.second << " " << (timestamp + time_offset) << "\n";
        }

        std::cout << "\nPredictions written to: " << config.output_file << "\n";
        std::cout << "Successfully predicted " << predictions.size() << " events.\n";
        if (config.denormalize_timestamps) {
            std::cout << "Timestamps denormalized (base time: " << read_result.base_time << ")\n";
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
