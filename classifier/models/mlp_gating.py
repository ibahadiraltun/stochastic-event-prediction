"""
MLP with Gating Mechanism for STEP Classification

This module implements a Multi-Layer Perceptron with a gating mechanism
that separately processes STEP temporal motif features and GNN features,
then combines them with learnable weights.
"""

import torch
import torch.nn as nn


class MLPWithGating(nn.Module):
    """
    MLP with gating mechanism for combining STEP and GNN features.

    The model processes two feature branches:
    - Earlier features (STEP temporal motif features)
    - Last features (GNN output)

    These branches are combined using learnable gating parameters (alpha, beta)
    before final classification.

    Args:
        total_dim: Total input dimension (STEP features + GNN features)
        last_dim: Dimension of the last features (GNN output)
        last_branch_dim: Hidden dimension for last branch (default: 16)
        hidden_dim: Hidden dimension for earlier branch (default: 64)
        dropout_p: Dropout probability (default: 0.5)
    """

    def __init__(self, total_dim, last_dim, last_branch_dim=16, hidden_dim=64, dropout_p=0.5):
        super(MLPWithGating, self).__init__()
        self.last_dim = last_dim
        earlier_dim = total_dim - last_dim

        # Earlier features branch (STEP motif features)
        self.earlier_fc = nn.Linear(earlier_dim, hidden_dim).float()
        nn.init.xavier_uniform_(self.earlier_fc.weight)
        self.earlier_act = nn.ReLU().float()
        self.earlier_dropout = nn.Dropout(p=dropout_p).float()
        self.earlier_proj = nn.Linear(hidden_dim, last_branch_dim).float()

        # Last features branch (GNN output)
        self.last_fc = nn.Linear(last_dim, last_branch_dim).float()
        nn.init.xavier_uniform_(self.last_fc.weight)
        self.last_act = nn.ReLU().float()
        self.last_dropout = nn.Dropout(p=dropout_p).float()

        # Gating parameters
        self.alpha = nn.Parameter(torch.tensor(1.0)).float()
        self.beta = nn.Parameter(torch.tensor(1.0)).float()

        # Final classification layer
        self.fc_final = nn.Linear(last_branch_dim, 1).float()
        nn.init.xavier_uniform_(self.fc_final.weight)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, total_dim)

        Returns:
            Logits tensor of shape (batch_size, 1)
        """
        # Split input into earlier (STEP) and last (GNN) features
        earlier = x[:, :x.size(1) - self.last_dim]
        last = x[:, -self.last_dim:]

        # Process earlier branch
        out_earlier = self.earlier_fc(earlier)
        out_earlier = self.earlier_act(out_earlier)
        out_earlier = self.earlier_dropout(out_earlier)
        out_earlier = self.earlier_proj(out_earlier)

        # Process last branch
        out_last = self.last_fc(last)
        out_last = self.last_act(out_last)
        out_last = self.last_dropout(out_last)

        # Combine with gating
        combined = self.alpha * out_earlier + self.beta * out_last

        # Final classification
        out = self.fc_final(combined)
        return out

    def get_gating_weights(self):
        """Return current gating weights (alpha, beta)"""
        return self.alpha.item(), self.beta.item()
