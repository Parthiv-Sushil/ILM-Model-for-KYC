# -*- coding: utf-8 -*-
"""
Spatial Feature Branch using MiniFASNet.
Extracts 128-dimensional spatial feature representations from cropped face images.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV2SE
from src.data_io import transform as trans
from src.utility import get_kernel, parse_model_name

MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE': MiniFASNetV1SE,
    'MiniFASNetV2SE': MiniFASNetV2SE
}


class SpatialFeatureExtractor(nn.Module):
    """
    Wrapper around MiniFASNet to extract 128-d embedding feature vectors
    and classification logits.
    """
    def __init__(self, model_path, device=None):
        super(SpatialFeatureExtractor, self).__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        model_name = os.path.basename(model_path)
        h_input, w_input, model_type, _ = parse_model_name(model_name)
        kernel_size = get_kernel(h_input, w_input)

        self.model = MODEL_MAPPING[model_type](conv6_kernel=kernel_size).to(self.device)
        self._load_weights(model_path)
        self.model.eval()

    def _load_weights(self, model_path):
        state_dict = torch.load(model_path, map_location=self.device)
        keys = iter(state_dict)
        first_layer_name = next(keys)
        if 'module.' in first_layer_name:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                new_state_dict[key[7:]] = value
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(state_dict)

    def extract_features(self, x):
        """
        Extract 128-d spatial embedding feature tensor from input batch (B, 3, H, W).
        Returns:
            embedding: (B, 128) feature tensor
            logits: (B, 3) raw prediction logits
        """
        out = self.model.conv1(x)
        out = self.model.conv2_dw(out)
        out = self.model.conv_23(out)
        out = self.model.conv_3(out)
        out = self.model.conv_34(out)
        out = self.model.conv_4(out)
        out = self.model.conv_45(out)
        out = self.model.conv_5(out)
        out = self.model.conv_6_sep(out)
        out = self.model.conv_6_dw(out)
        out = self.model.conv_6_flatten(out)
        if self.model.embedding_size != 512:
            out = self.model.linear(out)
        out = self.model.bn(out)
        embedding = out  # 128-d embedding
        logits = self.model.prob(out)
        return embedding, logits

    def forward(self, x):
        return self.extract_features(x)
