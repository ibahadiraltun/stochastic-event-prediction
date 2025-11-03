#!/usr/bin/env python3
"""
Generate example data for testing the STEP classifier.

Creates synthetic train and test files in the format:
label - feature1 feature2 feature3 ... featureN

The features simulate STEP temporal motif features + GNN output.
"""

import numpy as np
import argparse


def generate_features(n_samples, n_motif_features=10, positive_ratio=0.5, seed=42):
    """
    Generate synthetic features.

    Args:
        n_samples: Number of samples to generate
        n_motif_features: Number of STEP motif features
        positive_ratio: Ratio of positive samples
        seed: Random seed

    Returns:
        features: List of feature lists
        labels: List of labels
    """
    np.random.seed(seed)

    n_positive = int(n_samples * positive_ratio)
    n_negative = n_samples - n_positive

    features = []
    labels = []

    # Generate positive samples (label=1)
    for _ in range(n_positive):
        # STEP motif features (higher values for positive class)
        motif_feats = np.random.beta(5, 2, n_motif_features) * 0.1
        # GNN output (higher for positive class)
        gnn_feat = np.random.beta(5, 2, 1)[0] * 0.8 + 0.2

        features.append(list(motif_feats) + [gnn_feat])
        labels.append(1)

    # Generate negative samples (label=0)
    for _ in range(n_negative):
        # STEP motif features (lower values for negative class)
        motif_feats = np.random.beta(2, 5, n_motif_features) * 0.05
        # GNN output (lower for negative class)
        gnn_feat = np.random.beta(2, 5, 1)[0] * 0.3

        features.append(list(motif_feats) + [gnn_feat])
        labels.append(0)

    # Shuffle
    indices = np.random.permutation(n_samples)
    features = [features[i] for i in indices]
    labels = [labels[i] for i in indices]

    return features, labels


def write_feature_file(filename, features, labels):
    """Write features to file in the required format"""
    with open(filename, 'w') as f:
        for label, feats in zip(labels, features):
            feat_str = ' '.join([f"{x:.6f}" for x in feats])
            f.write(f"{label} - {feat_str}\n")


def main():
    parser = argparse.ArgumentParser(description='Generate example data for STEP classifier')
    parser.add_argument('--train-samples', type=int, default=1000,
                        help='Number of training samples (default: 1000)')
    parser.add_argument('--test-samples', type=int, default=200,
                        help='Number of test samples (default: 200)')
    parser.add_argument('--n-features', type=int, default=10,
                        help='Number of STEP motif features (default: 10)')
    parser.add_argument('--positive-ratio', type=float, default=0.5,
                        help='Ratio of positive samples (default: 0.5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    print("Generating example data...")
    print(f"  Train samples: {args.train_samples}")
    print(f"  Test samples:  {args.test_samples}")
    print(f"  Features:      {args.n_features} motif + 1 GNN = {args.n_features + 1} total")
    print(f"  Positive ratio: {args.positive_ratio}")

    # Generate train data
    train_features, train_labels = generate_features(
        args.train_samples,
        args.n_features,
        args.positive_ratio,
        args.seed
    )

    # Generate test data
    test_features, test_labels = generate_features(
        args.test_samples,
        args.n_features,
        args.positive_ratio,
        args.seed + 1
    )

    # Write to files
    write_feature_file('example_train.txt', train_features, train_labels)
    write_feature_file('example_test.txt', test_features, test_labels)

    print("\nFiles created:")
    print("  example_train.txt")
    print("  example_test.txt")
    print("\nYou can now run:")
    print("  python ../train.py example_train.txt example_test.txt")


if __name__ == "__main__":
    main()
