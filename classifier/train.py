#!/usr/bin/env python3
"""
STEP Classifier Training Script

Train a binary classifier on STEP temporal motif features combined with GNN features.
"""

import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim

from config import Config
from models.mlp_gating import MLPWithGating
from utils.data_loader import load_features, create_data_loaders, prepare_step_only_features
from utils.metrics import evaluate_model, print_classification_report, print_metrics


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch).view(-1)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


def train_and_evaluate(X_train, y_train, X_test, y_test, config, verbose=True):
    """
    Train and evaluate the classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        config: Configuration object
        verbose: Print progress (default: True)

    Returns:
        model: Trained model
        max_ap: Maximum average precision achieved
    """
    # Setup device
    device = torch.device("cuda:0" if config.USE_CUDA and torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")

    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        X_train, y_train, X_test, y_test,
        batch_size=config.BATCH_SIZE,
        device=device
    )

    # Get input dimension
    total_dim = len(X_train[0])

    # Create model
    model = MLPWithGating(
        total_dim=total_dim,
        last_dim=config.LAST_DIM,
        last_branch_dim=config.LAST_BRANCH_DIM,
        hidden_dim=config.HIDDEN_DIM,
        dropout_p=config.DROPOUT_P
    ).to(device)

    if verbose:
        print(f"\nModel: {total_dim}-dim input, {config.LAST_DIM}-dim GNN features")
        print(f"Gating weights (alpha, beta): {model.get_gating_weights()}")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # Training loop
    max_ap = 0.0
    for epoch in range(config.NUM_EPOCHS):
        avg_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate
        metrics, _ = evaluate_model(model, test_loader, device)
        epoch_ap = metrics['average_precision']

        if epoch_ap > max_ap:
            max_ap = epoch_ap

        if verbose:
            print(f"Epoch [{epoch + 1}/{config.NUM_EPOCHS}], "
                  f"Loss: {avg_loss:.4f}, "
                  f"Test AP: {epoch_ap:.4f}, "
                  f"Max AP: {max_ap:.4f}",
                  flush=True)

    # Final evaluation
    if verbose:
        print(f"\nFinal gating weights (alpha, beta): {model.get_gating_weights()}")

    metrics, predictions = evaluate_model(model, test_loader, device)

    if verbose:
        print_classification_report(predictions['labels'], predictions['predictions'])
        print(f"Maximum Average Precision: {max_ap:.4f}")

    return model, max_ap


def main():
    parser = argparse.ArgumentParser(description='STEP Classifier Training')
    parser.add_argument('train_file', help='Training features file')
    parser.add_argument('test_file', help='Test features file')
    parser.add_argument('--step-only', action='store_true',
                        help='Use only STEP features (set GNN feature to 0)')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=None, help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=None, help='Dropout probability')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress output')

    args = parser.parse_args()

    # Load configuration
    config = Config()

    # Override config with command line arguments
    if args.epochs is not None:
        config.NUM_EPOCHS = args.epochs
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.lr is not None:
        config.LEARNING_RATE = args.lr
    if args.hidden_dim is not None:
        config.HIDDEN_DIM = args.hidden_dim
    if args.dropout is not None:
        config.DROPOUT_P = args.dropout

    verbose = not args.quiet

    if verbose:
        print("=" * 60)
        print("STEP Classifier Training")
        print("=" * 60)
        print(f"\nConfiguration:")
        print(config)
        print()

    # Load data
    if verbose:
        print("Loading training data...")
    X_train, y_train = load_features(
        args.train_file,
        handle_inf=config.HANDLE_INF,
        min_value=config.MIN_VALUE
    )

    if verbose:
        print("Loading test data...")
    X_test, y_test = load_features(
        args.test_file,
        handle_inf=config.HANDLE_INF,
        min_value=config.MIN_VALUE
    )

    if verbose:
        print(f"Train: {len(X_train)} samples, {len(X_train[0])} features")
        print(f"Test:  {len(X_test)} samples, {len(X_test[0])} features")
        print(f"Label distribution (train): 0={y_train.count(0)}, 1={y_train.count(1)}")
        print()

    # Prepare STEP-only features if requested
    if args.step_only:
        if verbose:
            print("Using STEP-only mode (GNN features set to 0)")
        X_train = prepare_step_only_features(X_train)
        X_test = prepare_step_only_features(X_test)

    # Train and evaluate
    if verbose:
        print("\nStarting training...")
        print("-" * 60)

    model, max_ap = train_and_evaluate(X_train, y_train, X_test, y_test, config, verbose)

    if verbose:
        print("-" * 60)
        print(f"\nTraining complete!")
        print(f"Best Average Precision: {max_ap:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
