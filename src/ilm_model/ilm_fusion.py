# -*- coding: utf-8 -*-
"""
Multi-Modal Gated Feature Fusion Module.
Combines Spatial (128-d) + Temporal (64-d) + Liveness (8-d) + Frequency (16-d) features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusionLayer(nn.Module):
    """
    Learns dynamic gate weights for each feature modality branch.
    """
    def __init__(self, spatial_dim=128, temporal_dim=64, liveness_dim=8, freq_dim=16):
        super(GatedFusionLayer, self).__init__()
        self.total_dim = spatial_dim + temporal_dim + liveness_dim + freq_dim

        self.gate_fc = nn.Sequential(
            nn.Linear(self.total_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # 4 gating scalar weights for 4 modalities
            nn.Sigmoid()
        )

        self.spatial_proj = nn.Linear(spatial_dim, 64)
        self.temporal_proj = nn.Linear(temporal_dim, 64)
        self.liveness_proj = nn.Linear(liveness_dim, 32)
        self.freq_proj = nn.Linear(freq_dim, 32)

    def forward(self, f_spatial, f_temporal, f_liveness, f_freq):
        # Concatenate raw features to compute gating coefficients
        raw_concat = torch.cat([f_spatial, f_temporal, f_liveness, f_freq], dim=1)
        gates = self.gate_fc(raw_concat)  # (B, 4)

        g_s, g_t, g_l, g_f = gates[:, 0:1], gates[:, 1:2], gates[:, 2:3], gates[:, 3:4]

        p_s = self.spatial_proj(f_spatial) * g_s
        p_t = self.temporal_proj(f_temporal) * g_t
        p_l = self.liveness_proj(f_liveness) * g_l
        p_f = self.freq_proj(f_freq) * g_f

        fused_embedding = torch.cat([p_s, p_t, p_l, p_f], dim=1)  # 64+64+32+32 = 192-d
        return fused_embedding, gates


class ILMFusionClassifier(nn.Module):
    """
    Full Integrated Learning Model (ILM) fusion classifier.
    """
    def __init__(self, spatial_dim=128, temporal_dim=64, liveness_dim=8, freq_dim=16, num_classes=2):
        super(ILMFusionClassifier, self).__init__()
        self.gated_fusion = GatedFusionLayer(spatial_dim, temporal_dim, liveness_dim, freq_dim)

        self.classifier = nn.Sequential(
            nn.Linear(192, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, f_spatial, f_temporal, f_liveness, f_freq):
        fused_feat, gate_weights = self.gated_fusion(f_spatial, f_temporal, f_liveness, f_freq)
        logits = self.classifier(fused_feat)
        probs = F.softmax(logits, dim=1)
        return logits, probs, gate_weights
