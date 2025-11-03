"""
Evaluation metrics for STEP classifier.
"""

import torch
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    classification_report,
    average_precision_score
)


def evaluate_model(model, test_loader, device='cpu'):
    """
    Evaluate model on test set.

    Args:
        model: PyTorch model
        test_loader: Test DataLoader
        device: Device to run on (default: 'cpu')

    Returns:
        metrics: Dictionary with evaluation metrics
        predictions: Dictionary with probabilities, predictions, and labels
    """
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch).view(-1)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'average_precision': average_precision_score(all_labels, all_probs)
    }

    predictions = {
        'probabilities': all_probs,
        'predictions': all_preds,
        'labels': all_labels
    }

    return metrics, predictions


def print_classification_report(all_labels, all_preds):
    """Print detailed classification report"""
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))


def print_metrics(metrics):
    """Print evaluation metrics"""
    print(f"\nEvaluation Metrics:")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Precision:         {metrics['precision']:.4f}")
    print(f"  Recall:            {metrics['recall']:.4f}")
    print(f"  Average Precision: {metrics['average_precision']:.4f}")
