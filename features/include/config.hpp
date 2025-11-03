#pragma once

#include <string>
#include <stdexcept>

namespace step {

struct Config {
    std::string input_file;
    std::string output_file;
    int max_motif_length = 3;
    double delta_c = 0.0;  // 0 means auto-calculate from data

    // Shared memory configuration for Python predictor
    std::string shm_name = "/step_embedding_shm";
    size_t shm_size = 4096;  // Size in bytes
    bool use_python_predictor = true;
    int python_timeout_ms = 5000;  // Timeout waiting for Python response

    // Negative sampling configuration for classifier training
    bool enable_negative_sampling = false;
    double negative_sampling_ratio = 1.0;  // Number of negative samples per positive sample
    int random_seed = 42;  // Seed for reproducible negative sampling

    void validate() const {
        if (input_file.empty()) {
            throw std::invalid_argument("Input file path cannot be empty");
        }
        if (output_file.empty()) {
            throw std::invalid_argument("Output file path cannot be empty");
        }
        if (max_motif_length <= 0) {
            throw std::invalid_argument("Max motif length must be positive");
        }
        if (delta_c < 0.0) {
            throw std::invalid_argument("Delta C cannot be negative");
        }
        if (shm_name.empty()) {
            throw std::invalid_argument("Shared memory name cannot be empty");
        }
        if (shm_size < 1024) {
            throw std::invalid_argument("Shared memory size must be at least 1024 bytes");
        }
        if (python_timeout_ms <= 0) {
            throw std::invalid_argument("Python timeout must be positive");
        }
        if (negative_sampling_ratio < 0.0) {
            throw std::invalid_argument("Negative sampling ratio cannot be negative");
        }
    }
};

} // namespace step
