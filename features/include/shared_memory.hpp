#pragma once

#include <string>
#include <cstring>
#include <stdexcept>
#include <chrono>
#include <thread>

// POSIX shared memory
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace step {

/**
 * SharedMemoryChannel provides IPC between C++ and Python via POSIX shared memory
 *
 * Protocol:
 * - First 4 bytes: request flag (1 = C++ has written request, 0 = Python has read)
 * - Next 4 bytes: response flag (1 = Python has written response, 0 = C++ has read)
 * - Next 256 bytes: request data (event information: "node_u node_v timestamp")
 * - Next 8 bytes: response data (double probability value from neural network)
 *
 * Flow:
 * 1. C++ writes event info to request area, sets request_flag = 1
 * 2. Python reads event, runs neural network, sets request_flag = 0
 * 3. Python writes probability to response area, sets response_flag = 1
 * 4. C++ reads response, sets response_flag = 0
 *
 * Note:
 * - Python predictor receives raw events (not motifs)
 * - Timestamps are UNNORMALIZED (original timestamps from input file)
 * - Python returns independent probability predictions from its neural network
 */
class SharedMemoryChannel {
public:
    static constexpr size_t REQUEST_FLAG_OFFSET = 0;
    static constexpr size_t RESPONSE_FLAG_OFFSET = 4;
    static constexpr size_t REQUEST_DATA_OFFSET = 8;
    static constexpr size_t REQUEST_DATA_SIZE = 256;
    static constexpr size_t RESPONSE_DATA_OFFSET = REQUEST_DATA_OFFSET + REQUEST_DATA_SIZE;
    static constexpr size_t RESPONSE_DATA_SIZE = 8;  // sizeof(double)
    static constexpr size_t TOTAL_SIZE = RESPONSE_DATA_OFFSET + RESPONSE_DATA_SIZE;

    explicit SharedMemoryChannel(const std::string& name, size_t size)
        : shm_name_(name), shm_size_(size), shm_ptr_(nullptr), shm_fd_(-1) {
        if (shm_size_ < TOTAL_SIZE) {
            throw std::invalid_argument("Shared memory size too small");
        }
    }

    ~SharedMemoryChannel() {
        close();
    }

    // Delete copy constructor and assignment
    SharedMemoryChannel(const SharedMemoryChannel&) = delete;
    SharedMemoryChannel& operator=(const SharedMemoryChannel&) = delete;

    /**
     * Create and initialize shared memory (C++ side - producer)
     */
    void create() {
        // Remove any existing shared memory
        shm_unlink(shm_name_.c_str());

        // Create shared memory object
        shm_fd_ = shm_open(shm_name_.c_str(), O_CREAT | O_RDWR, 0666);
        if (shm_fd_ == -1) {
            throw std::runtime_error("Failed to create shared memory: " + std::string(strerror(errno)));
        }

        // Set size
        if (ftruncate(shm_fd_, shm_size_) == -1) {
            ::close(shm_fd_);
            shm_unlink(shm_name_.c_str());
            throw std::runtime_error("Failed to set shared memory size: " + std::string(strerror(errno)));
        }

        // Map to process address space
        shm_ptr_ = mmap(nullptr, shm_size_, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0);
        if (shm_ptr_ == MAP_FAILED) {
            ::close(shm_fd_);
            shm_unlink(shm_name_.c_str());
            throw std::runtime_error("Failed to map shared memory: " + std::string(strerror(errno)));
        }

        // Initialize flags to 0
        std::memset(shm_ptr_, 0, shm_size_);
    }

    /**
     * Open existing shared memory (Python side - consumer)
     */
    void open() {
        shm_fd_ = shm_open(shm_name_.c_str(), O_RDWR, 0666);
        if (shm_fd_ == -1) {
            throw std::runtime_error("Failed to open shared memory: " + std::string(strerror(errno)));
        }

        shm_ptr_ = mmap(nullptr, shm_size_, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0);
        if (shm_ptr_ == MAP_FAILED) {
            ::close(shm_fd_);
            throw std::runtime_error("Failed to map shared memory: " + std::string(strerror(errno)));
        }
    }

    /**
     * Send request and wait for response (C++ side)
     * @param event_info Event information as string "node_u node_v timestamp"
     * @param timeout_ms Timeout in milliseconds
     * @return Probability from Python neural network predictor
     */
    double request_prediction(const std::string& event_info, int timeout_ms) {
        if (shm_ptr_ == nullptr) {
            throw std::runtime_error("Shared memory not initialized");
        }

        // Write request data
        size_t copy_size = std::min(event_info.size(), REQUEST_DATA_SIZE - 1);
        std::memcpy(static_cast<char*>(shm_ptr_) + REQUEST_DATA_OFFSET,
                    event_info.c_str(), copy_size);
        static_cast<char*>(shm_ptr_)[REQUEST_DATA_OFFSET + copy_size] = '\0';

        // Set request flag
        set_flag(REQUEST_FLAG_OFFSET, 1);

        // Wait for response
        auto start = std::chrono::steady_clock::now();
        while (true) {
            if (get_flag(RESPONSE_FLAG_OFFSET) == 1) {
                // Read response
                double result;
                std::memcpy(&result, static_cast<char*>(shm_ptr_) + RESPONSE_DATA_OFFSET,
                           RESPONSE_DATA_SIZE);

                // Clear response flag
                set_flag(RESPONSE_FLAG_OFFSET, 0);

                return result;
            }

            // Check timeout
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start).count();
            if (elapsed > timeout_ms) {
                throw std::runtime_error("Timeout waiting for Python response");
            }

            // Small sleep to avoid busy waiting
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    /**
     * Check if there's a pending request (Python side)
     */
    bool has_request() const {
        return get_flag(REQUEST_FLAG_OFFSET) == 1;
    }

    /**
     * Read request data (Python side)
     */
    std::string read_request() {
        if (shm_ptr_ == nullptr) {
            throw std::runtime_error("Shared memory not initialized");
        }

        char buffer[REQUEST_DATA_SIZE];
        std::memcpy(buffer, static_cast<char*>(shm_ptr_) + REQUEST_DATA_OFFSET,
                   REQUEST_DATA_SIZE);
        buffer[REQUEST_DATA_SIZE - 1] = '\0';

        // Clear request flag
        set_flag(REQUEST_FLAG_OFFSET, 0);

        return std::string(buffer);
    }

    /**
     * Send response (Python side)
     */
    void send_response(double probability) {
        if (shm_ptr_ == nullptr) {
            throw std::runtime_error("Shared memory not initialized");
        }

        // Write response data
        std::memcpy(static_cast<char*>(shm_ptr_) + RESPONSE_DATA_OFFSET,
                   &probability, RESPONSE_DATA_SIZE);

        // Set response flag
        set_flag(RESPONSE_FLAG_OFFSET, 1);
    }

    /**
     * Close and cleanup shared memory
     */
    void close() {
        if (shm_ptr_ != nullptr) {
            munmap(shm_ptr_, shm_size_);
            shm_ptr_ = nullptr;
        }
        if (shm_fd_ != -1) {
            ::close(shm_fd_);
            shm_fd_ = -1;
        }
    }

    /**
     * Cleanup shared memory (C++ side should call this on exit)
     */
    void cleanup() {
        close();
        shm_unlink(shm_name_.c_str());
    }

private:
    std::string shm_name_;
    size_t shm_size_;
    void* shm_ptr_;
    int shm_fd_;

    void set_flag(size_t offset, int value) const {
        std::memcpy(static_cast<char*>(shm_ptr_) + offset, &value, sizeof(int));
    }

    int get_flag(size_t offset) const {
        int value;
        std::memcpy(&value, static_cast<char*>(shm_ptr_) + offset, sizeof(int));
        return value;
    }
};

} // namespace step
