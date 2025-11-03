# STEP: Sequential Temporal Event Predictor

A modern C++17 implementation of STEP (STochastic Event Predictor), a generative model for predicting future events in temporal networks by learning and extending network motifs.

## Overview

STEP unifies motif initiation and motif extension into a single generative loop. At each iteration, the algorithm:

1. Samples a global waiting time for the next event: `ΔT ~ Exponential(λ_global)`
2. Decides whether to start a new motif or extend an existing one
3. Updates the set of open motif instances

This approach captures both the local structure (motifs) and temporal dynamics of evolving networks.

## Project Structure

```
sequential/
├── include/             # Header files
│   ├── types.hpp        # Type definitions
│   ├── config.hpp       # Configuration structure
│   ├── hash_utils.hpp   # Hash functions for containers
│   ├── event_reader.hpp # Event data loading
│   ├── motif_processor.hpp  # Motif discovery and processing
│   ├── predictor.hpp    # Event prediction logic
│   └── scorer.hpp       # Evaluation metrics
├── src/
│   ├── main_refactored.cpp    # Main predictor (modern)
│   ├── scorer_refactored.cpp  # Evaluation tool (modern)
│   ├── main.cpp               # Legacy version
│   └── scorer.cpp             # Legacy scorer
├── CMakeLists.txt      # Build configuration
├── README.md           # This file
└── run.sh              # Convenience script
```

## Requirements

- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.14 or higher
- Standard C++ library

## Building

### Using run.sh (Recommended)

**Build the project:**
```bash
./run.sh build
```

**Clean build artifacts:**
```bash
./run.sh clean
```

**Clean and rebuild:**
```bash
./run.sh rebuild
```

**Show help:**
```bash
./run.sh help
```

### Manual Build

If you prefer to build manually without the script:

```bash
# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake ..

# Build
make -j$(nproc)

# Optionally install
make install
```

### Build Types

```bash
# Debug build (with symbols, no optimization)
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Release build (optimized, default)
cmake -DCMAKE_BUILD_TYPE=Release ..
```

## Usage

### Event Prediction

**Using run.sh:**
```bash
./run.sh predict <input_file> <output_file> <train_ratio> <prediction_size>
```

**Direct execution:**
```bash
./build/step-sequential <input_file> <output_file> <train_ratio> <prediction_size>
```

**Parameters:**
- `input_file`: Path to temporal network data (format: `node_u node_v timestamp`)
- `output_file`: Path to write predictions
- `train_ratio`: Fraction of data for training (e.g., 0.8 for 80%)
- `prediction_size`: Number of events to predict

**Example:**
```bash
./run.sh predict data/network.txt predictions.txt 0.8 100
```

### Evaluation

**Using run.sh:**
```bash
./run.sh score <ground_truth_file> <predictions_file> <train_ratio> <test_ratio>
```

**Direct execution:**
```bash
./build/scorer <ground_truth_file> <predictions_file> <train_ratio> <test_ratio>
```

**Parameters:**
- `ground_truth_file`: Original temporal network data
- `predictions_file`: Predicted events from step-sequential
- `train_ratio`: Same ratio used for training
- `test_ratio`: Fraction for testing (typically `1 - train_ratio`)

**Example:**
```bash
./run.sh score data/network.txt predictions.txt 0.8 0.2
```

### Run Example Workflow

To quickly test the system with example data:

```bash
./run.sh example
```

This command will:
- Build the project if not already built
- Generate sample network data
- Run prediction on the sample data
- Evaluate the predictions

## Input Format

The input file should contain one event per line in the format:
```
node_u node_v timestamp
```

**Example:**
```
1 2 1000
2 3 1005
1 3 1010
3 4 1020
```

**Notes:**
- Self-loops (edges where `u == v`) are automatically filtered
- Events are sorted by timestamp internally
- Duplicate events are removed
- Node IDs can be any integer

## Output Format

Predictions are written in the same format:
```
node_u node_v predicted_timestamp
```

**Timestamp Normalization:**

By default, timestamps in the output are denormalized (converted back to the original scale). During processing:
1. Input timestamps are normalized by subtracting the first event's timestamp
2. All internal processing uses normalized timestamps (starting from 0)
3. When writing predictions, the base timestamp is added back

This ensures predicted timestamps are in the same scale as the input data.

## Configuration

Default parameters (can be modified in [config.hpp](include/config.hpp)):

```cpp
train_ratio = 0.8;               // 80% training, 20% testing
prediction_size = 100;            // Predict 100 events
use_ratio = 0.5;                  // Use last 50% of training data for motifs
max_motif_length = 3;             // Maximum motif size
denormalize_timestamps = true;    // Add base timestamp back to predictions
```

**Timestamp Configuration:**
- Set `denormalize_timestamps = true` (default) to output timestamps in the original scale
- Set `denormalize_timestamps = false` to output normalized timestamps (starting from 0)

## Algorithm Details

### Motif Discovery

1. **Delta-C Calculation**: Determines the maximum time window for motif connectivity
2. **Motif Tracking**: Maintains open motifs that can be extended
3. **Transition Counting**: Records motif-to-motif transitions
4. **Inter-Event Times**: Calculates time distributions for each motif type

### Prediction Process

For each predicted event:

1. **Sample Global Time**: Draw from exponential distribution
2. **Choose Event Type**: Cold start (new motif) vs. hot (extend existing)
3. **Select Edge**:
   - **Cold**: Pick from previously unseen edge distribution
   - **Hot**: Pick based on motif transition probabilities
4. **Update State**: Add new motif instance or extend existing

## Evaluation Metrics

The scorer computes:

- **Precision**: `TP / (TP + FP)`
- **Recall**: `TP / (TP + FN)`
- **F1 Score**: `2 * (Precision * Recall) / (Precision + Recall)`

Where:
- TP = True Positives (correctly predicted edges in the future)
- FP = False Positives (incorrectly predicted edges in the future)
- FN = False Negatives (missed edges)

## Troubleshooting

### Build Errors

**Error: `CMAKE_CXX_STANDARD` is set to invalid value**
- Ensure your compiler supports C++17
- Update CMake to version 3.14+

**Error: Cannot find header files**
- Check that `include/` directory exists
- Verify CMakeLists.txt includes the correct paths

### Runtime Errors

**Error: Failed to open file**
- Check file path is correct
- Ensure file permissions allow reading
- Verify input format is correct

**Error: No valid events found**
- Check input file is not empty
- Verify format: `node_u node_v timestamp` (space-separated)
- Ensure no BOM or encoding issues

## License

## Citation

## Contact
