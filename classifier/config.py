"""
Configuration for STEP Classifier

This module contains default hyperparameters and settings for the classifier.
You can override these when calling the training script.
"""

class Config:
    """Default configuration for STEP classifier"""

    # Model architecture
    LAST_DIM = 1  # Dimension of GNN features (last feature)
    LAST_BRANCH_DIM = 16  # Hidden dimension for GNN branch
    HIDDEN_DIM = 64  # Hidden dimension for STEP features branch
    DROPOUT_P = 0.3  # Dropout probability

    # Training parameters
    BATCH_SIZE = 512
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Data processing
    HANDLE_INF = True  # Replace -inf with 0
    MIN_VALUE = 1e-12  # Minimum value for features

    # Device
    USE_CUDA = True  # Use CUDA if available

    # Evaluation
    THRESHOLD = 0.5  # Classification threshold

    def __repr__(self):
        return (
            f"Config(\n"
            f"  Model: hidden_dim={self.HIDDEN_DIM}, last_branch_dim={self.LAST_BRANCH_DIM}, "
            f"dropout={self.DROPOUT_P}\n"
            f"  Training: epochs={self.NUM_EPOCHS}, batch_size={self.BATCH_SIZE}, "
            f"lr={self.LEARNING_RATE}, wd={self.WEIGHT_DECAY}\n"
            f")"
        )
