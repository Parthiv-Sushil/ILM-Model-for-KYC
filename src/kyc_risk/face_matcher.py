# -*- coding: utf-8 -*-
"""
Lightweight Face Matcher for KYC Verification.
Computes cosine similarity between live face crops and reference ID document face images.
"""

import cv2
import numpy as np


class FaceMatcher:
    """
    Computes feature similarity score between reference ID photo and live face frame.
    """
    def __init__(self, target_size=(128, 128)):
        self.target_size = target_size

    def extract_descriptor(self, face_img):
        """
        Extract normalized spatial feature descriptor from face crop.
        """
        if face_img is None or face_img.size == 0:
            return None

        resized = cv2.resize(face_img, self.target_size)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized

        # Multi-scale histogram of gradients & color moments descriptor
        # 1. Color moment features (mean & std across BGR)
        if resized.ndim == 3:
            color_mean = np.mean(resized, axis=(0, 1)) / 255.0
            color_std = np.std(resized, axis=(0, 1)) / 255.0
        else:
            color_mean = np.array([np.mean(gray) / 255.0])
            color_std = np.array([np.std(gray) / 255.0])

        # 2. Gradient magnitude map
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad = np.sqrt(sobelx**2 + sobely**2)
        grad_resized = cv2.resize(grad, (16, 16)).flatten()
        grad_norm = grad_resized / (np.linalg.norm(grad_resized) + 1e-6)

        # 3. Spatial block intensity profile
        block_profile = cv2.resize(gray, (8, 8)).flatten() / 255.0

        descriptor = np.concatenate([color_mean, color_std, grad_norm, block_profile])
        norm = np.linalg.norm(descriptor)
        if norm > 0:
            descriptor = descriptor / norm
        return descriptor

    def compute_matching_score(self, id_face_img, live_face_img):
        """
        Computes cosine similarity match score in range [0.0, 1.0].
        If id_face_img is None, returns default high confidence simulation score (0.95).
        """
        if id_face_img is None:
            return 0.95  # Default simulated match for single-stream demos

        desc_id = self.extract_descriptor(id_face_img)
        desc_live = self.extract_descriptor(live_face_img)

        if desc_id is None or desc_live is None:
            return 0.0

        # Cosine similarity
        similarity = np.dot(desc_id, desc_live)
        # Rescale to [0, 1]
        score = float(np.clip(0.5 * (similarity + 1.0), 0.0, 1.0))
        return score
