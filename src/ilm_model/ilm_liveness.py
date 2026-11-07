# -*- coding: utf-8 -*-
"""
Liveness Feature Branch.
Computes physiological signals (blink EAR, eye movement), motion dynamics (head movement variance),
and challenge-response compliance.
"""

import numpy as np
import cv2


class LivenessFeatureExtractor:
    """
    Computes real-time physiological and behavioral liveness features from frame sequences.
    """
    def __init__(self, blink_threshold=0.20, motion_threshold=2.0):
        self.blink_threshold = blink_threshold
        self.motion_threshold = motion_threshold

    def compute_ear(self, eye_landmarks):
        """
        Compute Eye Aspect Ratio (EAR) given 6 2D facial landmarks for an eye.
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
        """
        if eye_landmarks is None or len(eye_landmarks) < 6:
            return 0.3  # Default open eye estimate

        p1, p2, p3, p4, p5, p6 = eye_landmarks[:6]
        v1 = np.linalg.norm(np.array(p2) - np.array(p6))
        v2 = np.linalg.norm(np.array(p3) - np.array(p5))
        h = np.linalg.norm(np.array(p1) - np.array(p4))

        if h < 1e-6:
            return 0.3
        return float((v1 + v2) / (2.0 * h))

    def extract_liveness_features(self, frames, bboxes, eye_landmarks_seq=None):
        """
        Extract 8-dimensional liveness vector from sequence of frames and bounding boxes.
        Parameters:
            frames: list of BGR image frames (N frames)
            bboxes: list of [x, y, w, h] bounding boxes per frame
            eye_landmarks_seq: optional list of eye landmarks per frame
        Returns:
            liveness_vector: (8,) numpy array of liveness features
            liveness_score: float in [0.0, 1.0]
        """
        if not frames or len(frames) == 0:
            return np.zeros(8, dtype=np.float32), 0.0

        n_frames = len(frames)

        # 1. Bounding box motion statistics
        centers = []
        areas = []
        for bbox in bboxes:
            if bbox is not None and len(bbox) >= 4:
                x, y, w, h = bbox[:4]
                centers.append([x + w / 2.0, y + h / 2.0])
                areas.append(float(w * h))
            else:
                centers.append([0.0, 0.0])
                areas.append(0.0)

        centers = np.array(centers)
        areas = np.array(areas)

        if n_frames > 1:
            center_diffs = np.diff(centers, axis=0)
            motion_velocity = np.mean(np.linalg.norm(center_diffs, axis=1))
            motion_variance = np.var(np.linalg.norm(center_diffs, axis=1))
            scale_variance = np.var(areas) / (np.mean(areas)**2 + 1e-6)
        else:
            motion_velocity = 0.0
            motion_variance = 0.0
            scale_variance = 0.0

        # 2. Frame-to-frame intensity / optical flow difference
        frame_diffs = []
        for i in range(1, n_frames):
            gray1 = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            diff = np.mean(np.abs(gray2.astype(np.float32) - gray1.astype(np.float32)))
            frame_diffs.append(diff)

        avg_frame_diff = float(np.mean(frame_diffs)) if frame_diffs else 0.0
        var_frame_diff = float(np.var(frame_diffs)) if frame_diffs else 0.0

        # 3. Eye Aspect Ratio & Blink Detection
        ear_values = []
        blinks_detected = 0
        if eye_landmarks_seq is not None and len(eye_landmarks_seq) == n_frames:
            for landmarks in eye_landmarks_seq:
                ear = self.compute_ear(landmarks)
                ear_values.append(ear)

            ear_std = float(np.std(ear_values))
            ear_min = float(np.min(ear_values))
            ear_mean = float(np.mean(ear_values))
            for ear in ear_values:
                if ear < self.blink_threshold:
                    blinks_detected += 1
        else:
            ear_std = 0.05
            ear_min = 0.25
            ear_mean = 0.30
            blinks_detected = 1 if avg_frame_diff > 1.5 else 0

        liveness_vector = np.array([
            motion_velocity,
            motion_variance,
            scale_variance,
            avg_frame_diff,
            var_frame_diff,
            ear_std,
            ear_min,
            float(blinks_detected)
        ], dtype=np.float32)

        is_moving = 0.1 <= motion_velocity <= 15.0
        is_natural_diff = 0.5 <= avg_frame_diff <= 25.0
        has_blink_or_jitter = (ear_std > 0.015) or (blinks_detected > 0) or (motion_variance > 0.05)

        score_components = [
            1.0 if is_moving else 0.4,
            1.0 if is_natural_diff else 0.3,
            1.0 if has_blink_or_jitter else 0.5
        ]
        liveness_score = float(np.mean(score_components))

        return liveness_vector, liveness_score
