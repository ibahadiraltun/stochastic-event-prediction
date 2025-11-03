# STEP: STochastic Event Prediction

A comprehensive framework for temporal link prediction utilizing temporal motifs, stochastic processes and bayesian inference.

## Overview

STEP (STochastic Event Prediction) is a framework that:
- Extracts temporal motif patterns from event sequences
- Generates realistic event sequences using motif-based sampling
- Computes feature vectors for machine learning pipelines
- Integrates with GNN-based predictors via shared memory
- Provides binary classification for link prediction tasks

This repository contains three main components:
1. **Sequential**: Event sequence generation using temporal motifs
2. **Features**: Feature vector extraction with GNN integration
3. **Classifier**: Binary classification combining STEP + GNN features

## Repository Structure

```
step/
├── sequential/          # Event sequence generation
│   ├── include/         # Header files (C++)
│   ├── src/             # Source code
│   ├── run.sh           # Build and run script
│   ├── README.md        # Sequential documentation
│   └── ...
│
├── features/            # Feature vector extraction with GNN integration
│   ├── include/         # Header files (C++)
│   ├── src/             # Source code
│   ├── python_predictor.py  # Example Python GNN predictor
│   ├── run.sh           # Build and run script
│   ├── README.md        # Features documentation
│   └── ...
│
├── classifier/          # Binary classification
│   ├── models/          # Neural network models
│   ├── utils/           # Data loading and metrics
│   ├── examples/        # Example data generation
│   ├── train.py         # Training script
│   ├── README.md        # Classifier documentation
│   └── ...
│
└── README.md            # This file
```

## Quick Start

### 1. Sequential: Generate Event Sequences

```bash
cd sequential
./run.sh build
./run.sh example
```

**Output**: Predicted event sequences based on temporal motif patterns

**Use case**: Generating realistic temporal network evolution, network simulation

### 2. Features: Extract Feature Vectors

```bash
cd features
./run.sh build
./run.sh example-python  # With Python GNN integration
```

**Output**: Feature matrix with STEP motif probabilities + GNN predictions

**Use case**: Preparing feature vectors for machine learning, combining STEP with GNN

### 3. Classifier: Train Binary Classifier

```bash
cd classifier
pip install -r requirements.txt
python examples/generate_example_data.py
python train.py examples/example_train.txt examples/example_test.txt
```

**Output**: Trained classifier with evaluation metrics (accuracy, AP, etc.)

**Use case**: Link prediction, event classification, temporal network analysis

## Components

### Sequential (`sequential/`)

**Purpose**: Generate event sequences using temporal motif sampling

**Key Features**:
- Motif-based stochastic event generation
- Exponential inter-event time distributions
- Configurable max motif length and time window (Delta-C)
- Efficient two-pass algorithm

**Algorithm**: Motif-based sampling with normalized posteriors

**Language**: C++17

**Input**: Event sequences (node_u, node_v, timestamp)

**Output**: Generated event sequences

**Documentation**: [sequential/README.md](sequential/README.md)

---

### Features (`features/`)

**Purpose**: Extract STEP feature vectors and integrate with Python GNN predictors

**Key Features**:
- Implements Algorithm 2 (STEP Feature Vectors)
- Computes normalized posterior probabilities for each motif type
- POSIX shared memory IPC with Python
- Negative sampling for binary classification
- Outputs labeled feature matrices

**Algorithm**: Algorithm 2 - Computation of STEP Feature Vectors

**Language**: C++17 (feature extraction) + Python 3.7+ (GNN predictor)

**Input**: Event sequences (node_u, node_v, timestamp)

**Output**: Feature matrix Φ ∈ R^(|E|×M) with labels

**Documentation**: [features/README.md](features/README.md)

---

### Classifier (`classifier/`)

**Purpose**: Binary classification combining STEP features and GNN output

**Key Features**:
- MLP with gating mechanism (separate STEP/GNN branches)
- Learnable gating weights (α, β) for optimal feature combination
- STEP-only mode for ablation studies
- Comprehensive evaluation metrics
- PyTorch implementation

**Algorithm**: Gated MLP with dual-branch architecture

**Language**: Python 3.7+ (PyTorch)

**Input**: Feature files (label - feature1 feature2 ... featureN)

**Output**: Trained classifier with predictions and metrics

**Documentation**: [classifier/README.md](classifier/README.md)

## Installation

### Requirements

**Sequential & Features (C++):**
- CMake 3.14+
- C++17 compiler (GCC 7+, Clang 5+, MSVC 2017+)
- POSIX-compliant OS (Linux, macOS)

**Features (Python GNN):**
- Python 3.7+
- NumPy, random (standard library)

**Classifier:**
- Python 3.7+
- PyTorch 2.0+
- NumPy 1.24+
- scikit-learn 1.3+

### Build C++ Components

```bash
# Sequential
cd sequential
./run.sh build

# Features
cd ../features
./run.sh build
```

### Install Python Dependencies

```bash
# Classifier
cd classifier
pip install -r requirements.txt
```

## Input Data Format

All components expect event sequences in the format:

```
node_u node_v timestamp
1 2 1000
2 3 1005
1 3 1010
```

- **node_u, node_v**: Integer node IDs
- **timestamp**: Integer timestamp (arbitrary units, must be sorted)
- **Self-loops**: Automatically filtered out (u ≠ v required)

## Troubleshooting

### Build Issues

**Error**: `CMake 3.14 or higher is required`
```bash
# macOS
brew install cmake

# Ubuntu/Debian
sudo apt-get install cmake
```

**Error**: `C++17 required`
```bash
# Update compiler: GCC 7+, Clang 5+, or MSVC 2017+
```

### Runtime Issues

**Error**: `Failed to open shared memory` (features)
- Ensure Python predictor starts before C++ process
- Check permissions: `ls -la /dev/shm/` (Linux)
- Try different shared memory name: `--shm-name /custom_name`

**Error**: `CUDA out of memory` (classifier)
- Reduce batch size: `--batch-size 256`
- Use CPU: Set `USE_CUDA = False` in config

### Data Issues

**Error**: `No valid events found in file`
- Check file format: space-separated `u v t`
- Ensure timestamps are integers and sorted
- Verify no empty lines or invalid characters

## Examples

### Example 1: Quick Test with Synthetic Data

```bash
# Generate and analyze synthetic network
cd features
./run.sh example

# View output
head example_output.csv
```

### Example 2: Full Pipeline with Python GNN

```bash
# Terminal 1: Start Python GNN predictor
cd features
python python_predictor.py

# Terminal 2: Extract features
./run.sh extract -i data.txt -o features.csv --negative

# Terminal 3: Train classifier
cd ../classifier
python train.py train.txt test.txt
```

### Example 3: STEP-Only Ablation Study

```bash
# Extract features without GNN
cd features
./run.sh extract -i data.txt -o features.csv --negative --no-python

# Train with STEP features only
cd ../classifier
python train.py train.txt test.txt --step-only

# Compare with full model
python train.py train.txt test.txt
```

## Publications

## Citation

## License

## Contributing

## Support

For questions, issues, or feature requests, please refer to the individual component README files:
- [Sequential README](sequential/README.md)
- [Features README](features/README.md)
- [Classifier README](classifier/README.md)

## Acknowledgments
