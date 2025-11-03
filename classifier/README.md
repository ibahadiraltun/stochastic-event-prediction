# STEP Classifier - Binary Classification with Temporal Motif Features

This directory implements a binary classifier that combines **STEP temporal motif features** with **GNN (Graph Neural Network) output** using a gated MLP architecture for link prediction and temporal network analysis.

## Overview

The STEP Classifier uses a Multi-Layer Perceptron with a gating mechanism to:
- Process STEP temporal motif features extracted from event sequences
- Incorporate GNN predictions as additional features
- Learn optimal combination weights (α, β) through gating
- Perform binary classification for link prediction tasks

### Key Features

- **Gated Architecture**: Separate processing branches for STEP and GNN features
- **Flexible Training**: Configurable hyperparameters and architecture
- **STEP-Only Mode**: Compare STEP features alone vs. STEP + GNN
- **Comprehensive Metrics**: Precision, recall, accuracy, and average precision
- **Modular Design**: Clean separation of model, data loading, and training logic

## Architecture

```
classifier/
├── models/
│   ├── __init__.py
│   └── mlp_gating.py          # MLP with gating mechanism
├── utils/
│   ├── __init__.py
│   ├── data_loader.py          # Data loading utilities
│   └── metrics.py              # Evaluation metrics
├── examples/
│   └── generate_example_data.py  # Generate synthetic data
├── config.py                   # Default configuration
├── train.py                    # Main training script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Model Architecture

### MLPWithGating

The model processes features in two branches:

```
Input Features: [STEP_motif_1, ..., STEP_motif_N, GNN_output]
                         ↓
            ┌────────────┴────────────┐
            │                         │
      STEP Branch                GNN Branch
            │                         │
      Linear(N → 64)            Linear(1 → 16)
      ReLU + Dropout            ReLU + Dropout
      Linear(64 → 16)                 │
            │                         │
            └─────────┬───────────────┘
                      ↓
              α * STEP + β * GNN
                      ↓
              Linear(16 → 1)
                      ↓
                   Sigmoid
                      ↓
                  Prediction
```

**Gating Parameters**:
- `α` (alpha): Weight for STEP features
- `β` (beta): Weight for GNN features
- Both are learned during training

## Installation

### Requirements

- Python 3.7+
- PyTorch 2.0+
- NumPy 1.24+
- scikit-learn 1.3+

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install torch numpy scikit-learn
```

## Input Format

The classifier expects feature files in the following format:

```
label - feature1 feature2 feature3 ... featureN
```

Where:
- **label**: Binary label (0 or 1)
- **features**: Space-separated feature values
  - First N-1 features: STEP temporal motif probabilities
  - Last feature: GNN output probability

### Example

```
1 - 0.823456 0.000000 0.654321 0.123456 0.789012
0 - 0.000000 0.111111 0.000000 0.222222 0.333333
1 - 0.456789 0.567890 0.678901 0.789012 0.890123
```

## Usage

### Basic Training

```bash
python train.py train_features.txt test_features.txt
```

### Training Options

```
positional arguments:
  train_file            Training features file
  test_file             Test features file

optional arguments:
  -h, --help            show this help message and exit
  --step-only           Use only STEP features (set GNN feature to 0)
  --epochs EPOCHS       Number of epochs (default: 100)
  --batch-size SIZE     Batch size (default: 512)
  --lr LR               Learning rate (default: 1e-3)
  --hidden-dim DIM      Hidden dimension (default: 64)
  --dropout P           Dropout probability (default: 0.3)
  --quiet               Suppress progress output
```

### Examples

#### Example 1: Generate and Train on Synthetic Data

```bash
# Generate example data
cd examples
python generate_example_data.py --train-samples 1000 --test-samples 200

# Train classifier
cd ..
python train.py examples/example_train.txt examples/example_test.txt
```

#### Example 2: STEP-Only Mode (No GNN Features)

```bash
python train.py train.txt test.txt --step-only
```

This sets the GNN feature to 0, allowing you to evaluate STEP features alone.

#### Example 3: Custom Hyperparameters

```bash
python train.py train.txt test.txt \
    --epochs 50 \
    --batch-size 256 \
    --lr 0.001 \
    --hidden-dim 128 \
    --dropout 0.5
```

#### Example 4: Quiet Mode (No Progress Output)

```bash
python train.py train.txt test.txt --quiet
```

## Configuration

Default hyperparameters are defined in [config.py](config.py):

```python
# Model architecture
LAST_DIM = 1              # GNN feature dimension
LAST_BRANCH_DIM = 16      # Hidden dim for GNN branch
HIDDEN_DIM = 64           # Hidden dim for STEP branch
DROPOUT_P = 0.3           # Dropout probability

# Training
BATCH_SIZE = 512
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Data processing
HANDLE_INF = True         # Replace -inf with 0
MIN_VALUE = 1e-12         # Minimum feature value
```

You can override these via command-line arguments or by modifying [config.py](config.py).

## Output

### Training Progress

```
Epoch [1/100], Loss: 0.6845, Test AP: 0.7234, Max AP: 0.7234
Epoch [2/100], Loss: 0.6201, Test AP: 0.7456, Max AP: 0.7456
...
Epoch [100/100], Loss: 0.2103, Test AP: 0.8923, Max AP: 0.8934
```

### Final Evaluation

```
Final gating weights (alpha, beta): (1.2345, 0.8765)

Classification Report:
              precision    recall  f1-score   support

         0.0     0.8912    0.8734    0.8822       100
         1.0     0.8789    0.8956    0.8872       100

    accuracy                        0.8845       200
   macro avg     0.8851    0.8845    0.8847       200
weighted avg     0.8851    0.8845    0.8847       200

Maximum Average Precision: 0.8934
```

The **gating weights** (α, β) indicate the relative importance learned for STEP features vs. GNN features.

## Integration with STEP Feature Vectors

To use this classifier with features extracted from `features/`:

1. **Extract features** using the feature extraction module:
   ```bash
   cd ../features
   ./run.sh extract -i data.txt -o features.csv --negative --neg-ratio 1.0
   ```

2. **Convert CSV to classifier format**:
   ```python
   import pandas as pd

   # Read CSV from feature extraction
   df = pd.read_csv('features.csv')

   # Extract features (all motif_* and gnn_output columns)
   feature_cols = [col for col in df.columns if col.startswith('motif_') or col == 'gnn_output']
   features = df[feature_cols].values
   labels = df['label'].values

   # Write in classifier format
   with open('classifier_input.txt', 'w') as f:
       for label, feat in zip(labels, features):
           feat_str = ' '.join([f"{x:.6f}" for x in feat])
           f.write(f"{int(label)} - {feat_str}\n")
   ```

3. **Train classifier**:
   ```bash
   cd ../classifier
   python train.py classifier_train.txt classifier_test.txt
   ```


## References

## License

## Contributing

## Support
