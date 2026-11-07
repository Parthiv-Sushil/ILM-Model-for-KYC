# -*- coding: utf-8 -*-
"""
Temporal Feature Branch using GRU/LSTM.
Captures temporal dynamics and motion patterns across consecutive video frames.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBranch(nn.Module):
    """
    Lightweight GRU/LSTM recurrent module for processing sequences of 128-d spatial features.
    Inputs:
        feature_sequence: Tensor of shape (B, T, 128) where T is sequence length (number of frames).
    Outputs:
        temporal_embedding: Tensor of shape (B, hidden_dim) representing temporal dynamics.
        temporal_score: Tensor of shape (B, 1) representing temporal liveness/consistency score.
    """
    def __init__(self, input_dim=128, hidden_dim=64, num_layers=1, rnn_type='gru', dropout=0.1):
        super(TemporalBranch, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type.lower()

        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=False
            )
        else:
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=False
            )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        self.temporal_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: Tensor of shape (B, T, input_dim)
        """
        if x.dim() == 2:
            # Single frame case: reshape to (B=1, T=1, input_dim)
            x = x.unsqueeze(1)

        batch_size, seq_len, _ = x.shape

        if self.rnn_type == 'lstm':
            rnn_out, (h_n, c_n) = self.rnn(x)
        else:
            rnn_out, h_n = self.rnn(x)

        # Temporal Attention aggregation across sequence length T
        attn_weights = self.attention(rnn_out)  # (B, T, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        context_vector = torch.sum(attn_weights * rnn_out, dim=1)  # (B, hidden_dim)

        temporal_score = self.temporal_head(context_vector)  # (B, 1)
        return context_vector, temporal_score
