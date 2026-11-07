# -*- coding: utf-8 -*-
"""
KYC Risk Scoring Engine.
Combines Anti-Spoofing, Liveness, Face Matching, and Document Verification scores
into an integrated Risk Score (0-100), Risk Level, and Decision Action.
"""

from src.kyc_risk.face_matcher import FaceMatcher


class KYCRiskEngine:
    """
    Configurable Risk Engine for Facial Recognition-Based Digital KYC Verification.
    Note: Weights and thresholds are experimental and subject to operational policy tuning.
    """
    def __init__(
        self,
        weight_antispoof=0.40,
        weight_liveness=0.25,
        weight_facematch=0.25,
        weight_doc=0.10,
        low_threshold=35.0,
        high_threshold=65.0,
        min_antispoof_approve=0.70,
        min_facematch_approve=0.70
    ):
        self.w_as = weight_antispoof
        self.w_live = weight_liveness
        self.w_fm = weight_facematch
        self.w_doc = weight_doc

        # Normalize weights to sum to 1.0
        total_w = self.w_as + self.w_live + self.w_fm + self.w_doc
        self.w_as /= total_w
        self.w_live /= total_w
        self.w_fm /= total_w
        self.w_doc /= total_w

        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.min_antispoof_approve = min_antispoof_approve
        self.min_facematch_approve = min_facematch_approve

        self.face_matcher = FaceMatcher()

    def evaluate_kyc(
        self,
        antispoof_score,
        liveness_score,
        face_match_score=None,
        doc_valid_score=1.0,
        id_face_img=None,
        live_face_img=None
    ):
        """
        Evaluate full KYC verification payload.
        Parameters:
            antispoof_score: float [0, 1] (Probability of Real Face)
            liveness_score: float [0, 1] (Liveness confidence)
            face_match_score: optional float [0, 1]
            doc_valid_score: float [0, 1] (Document authenticity result)
            id_face_img: optional BGR image of reference ID photo
            live_face_img: optional BGR image of live face crop
        Returns:
            dict containing:
                risk_score: float [0, 100]
                risk_level: 'LOW' | 'MEDIUM' | 'HIGH'
                decision: 'APPROVE' | 'MANUAL REVIEW' | 'REJECT'
                breakdown: dict of individual weighted scores
        """
        if face_match_score is None:
            if id_face_img is not None and live_face_img is not None:
                face_match_score = self.face_matcher.compute_matching_score(id_face_img, live_face_img)
            else:
                face_match_score = 0.95  # Default simulated match

        antispoof_score = float(max(0.0, min(1.0, antispoof_score)))
        liveness_score = float(max(0.0, min(1.0, liveness_score)))
        face_match_score = float(max(0.0, min(1.0, face_match_score)))
        doc_valid_score = float(max(0.0, min(1.0, doc_valid_score)))

        # Composite Trust Index [0.0, 1.0]
        trust_index = (
            self.w_as * antispoof_score +
            self.w_live * liveness_score +
            self.w_fm * face_match_score +
            self.w_doc * doc_valid_score
        )

        # Risk Score [0, 100] (Higher score indicates HIGHER risk)
        risk_score = float((1.0 - trust_index) * 100.0)

        # Risk Level Mapping
        if risk_score < self.low_threshold:
            risk_level = "LOW"
        elif risk_score < self.high_threshold:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        # Action Decision Logic
        if (
            risk_score < self.low_threshold and
            antispoof_score >= self.min_antispoof_approve and
            face_match_score >= self.min_facematch_approve and
            doc_valid_score >= 0.5
        ):
            decision = "APPROVE"
        elif risk_score >= self.high_threshold or antispoof_score < 0.40 or face_match_score < 0.40:
            decision = "REJECT"
        else:
            decision = "MANUAL REVIEW"

        return {
            "risk_score": round(risk_score, 2),
            "risk_level": risk_level,
            "decision": decision,
            "breakdown": {
                "antispoof_score": round(antispoof_score, 4),
                "liveness_score": round(liveness_score, 4),
                "face_match_score": round(face_match_score, 4),
                "doc_valid_score": round(doc_valid_score, 4),
                "trust_index": round(trust_index, 4)
            }
        }
