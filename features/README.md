# STEP Feature Vectors - Feature Extraction with Neural Network Integration

This module computes temporal motif-based feature vectors by calculating **normalized posterior probabilities** for motif transitions in event sequences. It identifies recurring temporal patterns (motifs) and quantifies the likelihood of each motif type occurring given the historical context, optionally integrating with Python-based neural network predictors via shared memory for enhanced predictions.

## Overview

The STEP feature extraction system:
- Extracts feature vectors from chronological event sequences using temporal motifs
- Computes normalized posterior probabilities for each motif type
- Provides inter-process communication (IPC) with Python for neural network augmentation
- Supports negative sampling for classifier training (positive/negative labels)
- Outputs labeled feature matrices suitable for machine learning pipelines

### Key Differences from `sequential/`

| Aspect | `sequential/` | `features/` |
|--------|--------------|-------------|
| Purpose | Event prediction | Feature extraction |
| Output | Predicted events | Feature matrix Φ ∈ R^(\|E\|×M) |
| Algorithm | Motif-based sampling | Algorithm 2 (feature vectors) |
| Python Integration | ❌ No | ✅ Yes (via shared memory) |
| Use Case | Generate event sequences | Extract feature vectors for ML models |

## Architecture

```
features/
├── include/
│   ├── types.hpp              # Type definitions
│   ├── config.hpp             # Configuration management
│   ├── event_reader.hpp       # Data loading and preprocessing
│   ├── hash_utils.hpp         # Hash functions for STL containers
│   ├── feature_extractor.hpp  # Core Algorithm 2 implementation
│   └── shared_memory.hpp      # POSIX shared memory IPC
├── src/
│   └── main.cpp               # Main executable
├── CMakeLists.txt             # Build configuration
├── run.sh                     # Convenience script
├── python_predictor.py        # Example Python predictor
├── example_data.txt           # Sample input data
└── README.md                  # This file
```

## Building

### Requirements

- CMake 3.14+
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- Python 3.6+ (optional, for neural network integration)

### Quick Build

```bash
./run.sh build
```

### Manual Build

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

The executable will be created at `build/step-embedding`.

## Usage

### Basic Usage (Without Python)

```bash
./run.sh extract -i input_data.txt -o features.csv --no-python
```

### With Python Neural Network Integration

1. **Start Python predictor first:**
```bash
python3 python_predictor.py
```

2. **Run feature extraction (in another terminal):**
```bash
./run.sh extract -i input_data.txt -o features.csv
```

### Command-Line Options

```
Options:
  -i <file>         Input file path (required)
  -o <file>         Output file path (required)
  -m <int>          Max motif length (default: 3)
  -d <double>       Delta C time window (default: auto-calculate)
  --negative        Enable negative sampling for classifier
  --neg-ratio <r>   Negative sampling ratio (default: 1.0)
  --seed <int>      Random seed for negative sampling (default: 42)
  --no-python       Disable Python predictor integration
  --shm-name <s>    Shared memory name (default: /step_embedding_shm)
  --timeout <ms>    Python timeout in milliseconds (default: 5000)
  -h, --help        Show help message
```

### Examples

#### Example 1: Basic Feature Extraction

```bash
./run.sh example
```

This runs the example without Python integration.

#### Example 2: With Python Neural Network (Recommended)

```bash
./run.sh example-python
```

This automatically starts Python predictor, runs feature extraction, and shows results. The Python predictor receives raw events and returns probability predictions.

#### Example 3: With Custom Parameters

```bash
./run.sh extract \
    -i data/network_events.txt \
    -o results/features.csv \
    -m 4 \
    -d 100.0
```

#### Example 4: With Negative Sampling for Classification

```bash
./run.sh extract \
    -i data.txt \
    -o features_classifier.csv \
    --negative \
    --neg-ratio 1.0 \
    --seed 42 \
    --no-python
```

This generates equal numbers of positive and negative samples with labels for classifier training.

#### Example 5: Manual Python Integration (Two Terminals)

If you prefer manual control:

```bash
# Terminal 1: Start Python predictor
python3 python_predictor.py

# Terminal 2: Run feature extraction
./run.sh extract -i data.txt -o features.csv
```

#### Example 6: Full Pipeline (Negative Sampling + Python NN)

```bash
# Terminal 1: Start Python predictor
python3 python_predictor.py

# Terminal 2: Extract features with negative sampling
./run.sh extract \
    -i data.txt \
    -o features_full.csv \
    --negative \
    --neg-ratio 1.0
```

## Input Format

Space-separated text file with one event per line:

```
node_u node_v timestamp
1 2 1000
2 3 1005
1 3 1010
3 4 1020
```

- `node_u`, `node_v`: Integer node IDs
- `timestamp`: Integer timestamp (arbitrary units)
- Self-loops (u == v) are automatically filtered out

## Output Format

CSV file with labeled feature vectors for each event:

```csv
event_index,node_u,node_v,timestamp,label,motif_01,motif_0112,motif_0123,...
0,1,2,0,1,0.000000,0.000000,0.000000,...
1,2,3,5,1,0.823456,0.000000,0.000000,...
2,1,3,10,1,0.000000,0.654321,0.000000,...
15,3,7,45,0,0.000000,0.000000,0.000000,...
```

- **event_index**: Sequential event number
- **node_u, node_v**: Event edge nodes
- **timestamp**: Normalized timestamp
- **label**: 1 = positive (real event), 0 = negative (sampled event)
- **motif_XXX**: Feature values for each motif type
- **gnn_output**: (optional) Probability from Python GNN/neural network if enabled

## Python Integration

### Shared Memory Protocol

Communication uses POSIX shared memory with the following structure:

```
Offset | Size | Description
-------|------|------------
0      | 4    | Request flag (1 = new request, 0 = read)
4      | 4    | Response flag (1 = new response, 0 = read)
8      | 256  | Request data (event: "node_u node_v timestamp")
264    | 8    | Response data (double probability)
```

**Important**: Python receives **raw events** (not motif information). The neural network should be trained independently on the same dataset and provide its own probability predictions.

### Communication Flow

1. **C++ → Python**: Write event info "node_u node_v timestamp", set request_flag = 1
2. **Python**: Read event, run neural network, set request_flag = 0
3. **Python → C++**: Write probability, set response_flag = 1
4. **C++**: Read response, set response_flag = 0

### Custom Python Predictor

Replace the example predictor with your own neural network:

```python
def predict(self, event_info):
    """
    Args:
        event_info: String "node_u node_v timestamp"

    Returns:
        Predicted probability (float)
    """
    # Parse input
    parts = event_info.split()
    node_u, node_v, timestamp = int(parts[0]), int(parts[1]), int(parts[2])

    # YOUR NEURAL NETWORK HERE
    # The network should be trained on the same dataset
    features = extract_features(node_u, node_v, timestamp)
    prediction = neural_network.predict(features)

    return prediction[0]
```

### Request Format

The C++ process sends raw events in the format:
```
"node_u node_v timestamp"
```

Example:
```
"1 2 1005"
```

Where:
- `node_u`: Source node ID
- `node_v`: Target node ID
- `timestamp`: Event timestamp (**unnormalized** - original timestamp from input file)

**Note**: The Python neural network receives events, not motifs. Timestamps are sent **unnormalized** (with base_time added back) so Python sees the original timestamps from your dataset.

## Motif Encoding

Motifs are encoded using canonical node indexing:

| Motif Structure | Code | Description |
|----------------|------|-------------|
| (0,1) | `01` | Single edge |
| (0,1)→(1,2) | `0112` | Chain |
| (0,1)→(0,2) | `0102` | Star |
| (0,1)→(1,0) | `0110` | Reciprocal |
| (0,1)→(1,2)→(2,3) | `011223` | 3-chain |
| (0,1)→(1,2)→(0,2) | `011202` | Triangle |

Nodes are indexed in order of first appearance in the motif.

## Troubleshooting

### Build Errors

**Error**: `CMake 3.14 or higher is required`
```bash
# macOS
brew install cmake

# Ubuntu/Debian
sudo apt-get install cmake
```

**Error**: `C++17 required`
```bash
# Update compiler
# GCC 7+, Clang 5+, or MSVC 2017+
```

### Runtime Errors

**Error**: `Failed to open shared memory`
- Ensure Python predictor is started **before** C++ process
- Check shared memory permissions: `ls -la /dev/shm/` (Linux)
- Try different shared memory name: `--shm-name /custom_shm_name`

**Error**: `Timeout waiting for Python response`
- Increase timeout: `--timeout 10000` (10 seconds)
- Check Python process is running: `ps aux | grep python_predictor`
- Verify Python predictor is connected (check stdout)

**Error**: `No valid events found in file`
- Check input file format (space-separated: u v timestamp)
- Ensure file exists and is readable
- Verify timestamps are integers

### Memory Issues

**Error**: `std::bad_alloc` or crashes
- Reduce `max_motif_length`
- Process smaller subsets of data
- Increase system memory limits

## References

## License

## Contributing

## Support
