# -*- coding: utf-8 -*-
"""
ILM Unified Predictor Pipeline.
Orchestrates Face Detection, Spatial, Temporal, Liveness, Frequency Feature Extraction,
and Multi-Modal Fusion Prediction.
"""

import os
import torch
import cv2
import numpy as np
from src.anti_spoof_predict import Detection
from src.generate_patches import CropImage
from src.utility import parse_model_name
from src.ilm_model.ilm_spatial import SpatialFeatureExtractor
from src.ilm_model.ilm_temporal import TemporalBranch
from src.ilm_model.ilm_liveness import LivenessFeatureExtractor
from src.ilm_model.ilm_frequency import FrequencyFeatureExtractor
from src.ilm_model.ilm_fusion import ILMFusionClassifier


class ILMPredictor:
    """
    Unified Integrated Learning Model Predictor for video sequence processing.
    """
    def __init__(self, model_dir="./resources/anti_spoof_models", device_id=0):
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        self.detector = Detection()
        self.image_cropper = CropImage()
        self.model_dir = model_dir

        # Initialize spatial extractors for available pretrained models
        self.spatial_extractors = {}
        if os.path.exists(model_dir):
            for m_name in os.listdir(model_dir):
                if m_name.endswith(".pth"):
                    m_path = os.path.join(model_dir, m_name)
                    self.spatial_extractors[m_name] = SpatialFeatureExtractor(m_path, device=self.device)

        # Initialize ILM branches
        self.temporal_branch = TemporalBranch(input_dim=128, hidden_dim=64).to(self.device)
        self.temporal_branch.eval()

        self.liveness_extractor = LivenessFeatureExtractor()
        self.freq_extractor = FrequencyFeatureExtractor()

        self.fusion_classifier = ILMFusionClassifier(
            spatial_dim=128, temporal_dim=64, liveness_dim=8, freq_dim=16
        ).to(self.device)
        self.fusion_classifier.eval()

    def process_frame(self, frame):
        """
        Process a single image frame: detect face bbox and extract spatial/frequency features.
        Returns:
            bbox: [x, y, w, h] or None
            spatial_feat: (128,) numpy vector or None
            baseline_prob: float (real face probability from original baseline model ensemble)
            freq_feat: (16,) numpy vector
        """
        try:
            bbox = self.detector.get_bbox(frame)
        except Exception:
            return None, None, 0.0, np.zeros(16, dtype=np.float32)

        if bbox is None or len(bbox) < 4 or bbox[2] <= 0 or bbox[3] <= 0:
            return None, None, 0.0, np.zeros(16, dtype=np.float32)

        # 1. Baseline prediction & Spatial Feature Extraction across models
        spatial_feats = []
        baseline_preds = np.zeros(3)

        for m_name, extractor in self.spatial_extractors.items():
            h_input, w_input, model_type, scale = parse_model_name(m_name)
            param = {
                "org_img": frame,
                "bbox": bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True if scale is not None else False
            }
            crop_img = self.image_cropper.crop(**param)

            # Preprocessing to Tensor
            img_t = torch.from_numpy(crop_img.transpose((2, 0, 1))).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat, logits = extractor(img_t)
                prob = torch.softmax(logits, dim=1).cpu().numpy()[0]

            spatial_feats.append(feat.cpu().numpy()[0])
            baseline_preds += prob

        # Ensemble baseline probability (label 1 = Real Face)
        avg_baseline_preds = baseline_preds / max(len(self.spatial_extractors), 1)
        real_prob_baseline = float(avg_baseline_preds[1])

        # Mean spatial embedding (128-d)
        mean_spatial_feat = np.mean(spatial_feats, axis=0) if spatial_feats else np.zeros(128, dtype=np.float32)

        # 2. Frequency Feature Extraction on Face Crop
        freq_feat = self.freq_extractor.extract_features(frame)

        return bbox, mean_spatial_feat, real_prob_baseline, freq_feat

    def predict_sequence(self, frame_sequence):
        """
        Process a multi-frame sequence through the full ILM integrated pipeline.
        Parameters:
            frame_sequence: List of BGR frames (e.g. 5-30 frames)
        Returns:
            dict containing:
                ilm_real_prob: float [0, 1]
                baseline_real_prob: float [0, 1]
                liveness_score: float [0, 1]
                temporal_score: float [0, 1]
                bbox: face bbox from last valid frame
                gate_weights: modal importance weights
        """
        if not frame_sequence:
            return {
                "ilm_real_prob": 0.0,
                "baseline_real_prob": 0.0,
                "liveness_score": 0.0,
                "temporal_score": 0.0,
                "bbox": None,
                "gate_weights": [0.25, 0.25, 0.25, 0.25]
            }

        spatial_seq = []
        baseline_probs = []
        freq_seq = []
        bboxes = []
        last_valid_bbox = None

        for frame in frame_sequence:
            bbox, s_feat, b_prob, f_feat = self.process_frame(frame)
            if bbox is not None and s_feat is not None:
                spatial_seq.append(s_feat)
                baseline_probs.append(b_prob)
                freq_seq.append(f_feat)
                bboxes.append(bbox)
                last_valid_bbox = bbox

        if not spatial_seq:
            return {
                "ilm_real_prob": 0.0,
                "baseline_real_prob": 0.0,
                "liveness_score": 0.0,
                "temporal_score": 0.0,
                "bbox": None,
                "gate_weights": [0.25, 0.25, 0.25, 0.25]
            }

        # 1. Temporal Branch Pass
        spatial_tensor = torch.tensor(np.array(spatial_seq), dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            t_context, t_score_tensor = self.temporal_branch(spatial_tensor)
            temporal_score = float(t_score_tensor.cpu().numpy()[0, 0])

        # 2. Liveness Branch Pass
        liveness_vector, liveness_score = self.liveness_extractor.extract_liveness_features(
            frame_sequence, bboxes
        )

        # 3. Aggregated Spatial and Frequency Vectors
        avg_spatial = torch.tensor(np.mean(spatial_seq, axis=0), dtype=torch.float32).unsqueeze(0).to(self.device)
        avg_freq = torch.tensor(np.mean(freq_seq, axis=0), dtype=torch.float32).unsqueeze(0).to(self.device)
        liveness_tensor = torch.tensor(liveness_vector, dtype=torch.float32).unsqueeze(0).to(self.device)

        # 4. Multi-Modal Gated Fusion Pass
        with torch.no_grad():
            logits, probs, gates = self.fusion_classifier(avg_spatial, t_context, liveness_tensor, avg_freq)
            gate_weights = gates.cpu().numpy()[0].tolist()

        # Heuristic ensemble combining spatial baseline, temporal, liveness, and fusion network
        baseline_avg_prob = float(np.mean(baseline_probs))
        ilm_prob = float(probs[0, 1].cpu().numpy())  # Index 1 = Real Face probability

        # Combined integrated score
        integrated_real_prob = (
            0.40 * baseline_avg_prob +
            0.30 * ilm_prob +
            0.15 * liveness_score +
            0.15 * temporal_score
        )
        integrated_real_prob = float(np.clip(integrated_real_prob, 0.0, 1.0))

        return {
            "ilm_real_prob": integrated_real_prob,
            "baseline_real_prob": baseline_avg_prob,
            "liveness_score": liveness_score,
            "temporal_score": temporal_score,
            "bbox": last_valid_bbox,
            "gate_weights": gate_weights
        }
