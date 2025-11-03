#pragma once

#include <string>
#include <stdexcept>

namespace step {

struct Config {
    std::string input_file;
    std::string output_file;
    double train_ratio = 0.8;
    int prediction_size = 100;
    double use_ratio = 0.5;
    int max_motif_length = 3;
    bool denormalize_timestamps = true;  // Add base timestamp back to predictions

    void validate() const {
        if (input_file.empty()) {
            throw std::invalid_argument("Input file path cannot be empty");
        }
        if (output_file.empty()) {
            throw std::invalid_argument("Output file path cannot be empty");
        }
        if (train_ratio <= 0.0 || train_ratio > 1.0) {
            throw std::invalid_argument("Train ratio must be between 0 and 1 (inclusive)");
        }
        if (prediction_size <= 0) {
            throw std::invalid_argument("Prediction size must be positive");
        }
        if (use_ratio <= 0.0 || use_ratio > 1.0) {
            throw std::invalid_argument("Use ratio must be between 0 and 1 (inclusive)");
        }
        if (max_motif_length <= 0) {
            throw std::invalid_argument("Max motif length must be positive");
        }
    }
};

} // namespace step
