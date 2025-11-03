"""
Data loading utilities for STEP classifier.

Handles loading feature files in the format:
label - feature1 feature2 feature3 ...
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def load_features(file_path, handle_inf=True, min_value=1e-12):
    """
    Load features from file.

    File format:
        label - feature1 feature2 feature3 ...

    Args:
        file_path: Path to feature file
        handle_inf: Replace -inf with 0 (default: True)
        min_value: Minimum value for features to avoid numerical issues (default: 1e-12)

    Returns:
        X: Feature matrix (list of lists)
        y: Labels (list)
    """
    X = []
    y = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Split label and features
            parts = line.split(' - ')
            if len(parts) != 2:
                raise ValueError(f"Invalid line format: {line}")

            # Parse label
            label = int(parts[0])
            y.append(label)

            # Parse features
            feature_str = parts[1].strip()
            features = []
            for x in feature_str.split():
                # Handle -inf values
                if handle_inf and x == '-inf':
                    features.append(0.0)
                else:
                    features.append(float(x))

            # Apply minimum value threshold
            if min_value is not None:
                features = [max(x, min_value) if x > 0 else x for x in features]

            X.append(features)

    return X, y


def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=512, device='cpu'):
    """
    Create PyTorch DataLoaders for training and testing.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        batch_size: Batch size (default: 512)
        device: Device to move tensors to (default: 'cpu')

    Returns:
        train_loader: Training DataLoader
        test_loader: Test DataLoader
    """
    # Convert to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Convert to tensors
    X_train_tensor = torch.from_numpy(X_train).float().to(device)
    y_train_tensor = torch.from_numpy(y_train).float().to(device)
    X_test_tensor = torch.from_numpy(X_test).float().to(device)
    y_test_tensor = torch.from_numpy(y_test).float().to(device)

    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def prepare_step_only_features(X, placeholder_value=0.0):
    """
    Prepare features for STEP-only mode by replacing GNN features with placeholder.

    This is useful for comparing STEP features alone vs. STEP + GNN features.

    Args:
        X: Original features (list of lists)
        placeholder_value: Value to replace GNN feature with (default: 0.0)

    Returns:
        X_step_only: Modified features with GNN replaced
    """
    return [features[:-1] + [placeholder_value] for features in X]
