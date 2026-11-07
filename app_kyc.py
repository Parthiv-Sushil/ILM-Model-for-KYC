# -*- coding: utf-8 -*-
"""
Streamlit Web Interface for Digital KYC Verification & Live Anti-Spoofing Check.
Provides an interactive browser interface for real-time camera checking and video evaluation.
"""

import os
import cv2
import time
import numpy as np
import streamlit as st
from PIL import Image

from src.ilm_model.ilm_predictor import ILMPredictor
from src.kyc_risk.kyc_engine import KYCRiskEngine
from demo_kyc import draw_kyc_dashboard


st.set_page_config(
    page_title="Digital KYC Verification - ILM Model",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Digital KYC Verification - Integrated Learning Model (ILM)")
st.markdown("""
**Research Prototype**: Deepfake & Presentation Attack Detection for Digital KYC Verification.
*Integrates Spatial (MiniFASNet), Temporal (GRU), Liveness (EAR & Motion), and Frequency (2D FFT) Feature Fusion.*
""")

# Sidebar Controls
st.sidebar.header("⚙️ Configuration")
mode = st.sidebar.radio("Select Verification Mode", ["Webcam Live Check", "Upload Video File", "Sample Images Evaluation"])

id_file = st.sidebar.file_uploader("Upload Reference ID Card Photo (Optional)", type=["jpg", "jpeg", "png"])
id_face_img = None
if id_file is not None:
    id_bytes = np.asarray(bytearray(id_file.read()), dtype=np.uint8)
    id_face_img = cv2.imdecode(id_bytes, 1)
    st.sidebar.image(id_file, caption="Reference ID Photo", width=150)

@st.cache_resource
def load_models():
    predictor = ILMPredictor()
    engine = KYCRiskEngine()
    return predictor, engine

predictor, kyc_engine = load_models()

col1, col2 = st.columns([2, 1])

if mode == "Webcam Live Check":
    col1.subheader("📹 Live Camera Feed")
    run_cam = col1.checkbox("Start Camera Stream", value=True)
    frame_placeholder = col1.empty()
    status_placeholder = col2.empty()

    if run_cam:
        cap = cv2.VideoCapture(0)
        frame_buffer = []
        fps_start = time.time()
        frame_count = 0

        while cap.isOpened() and run_cam:
            ret, frame = cap.read()
            if not ret:
                col1.error("Unable to access camera feed.")
                break

            frame_count += 1
            frame_buffer.append(frame.copy())
            if len(frame_buffer) > 5:
                frame_buffer.pop(0)

            ilm_metrics = predictor.predict_sequence(frame_buffer)
            kyc_result = kyc_engine.evaluate_kyc(
                antispoof_score=ilm_metrics["ilm_real_prob"],
                liveness_score=ilm_metrics["liveness_score"],
                id_face_img=id_face_img,
                live_face_img=frame
            )

            elapsed_fps = frame_count / max(time.time() - fps_start, 1e-5)
            annotated = draw_kyc_dashboard(
                frame.copy(), kyc_result, ilm_metrics, bbox=ilm_metrics["bbox"], fps=elapsed_fps
            )

            # Convert BGR to RGB for Streamlit display
            rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

            with status_placeholder.container():
                st.markdown("### 📊 Verification Status")
                dec = kyc_result["decision"]
                if dec == "APPROVE":
                    st.success(f"**DECISION: {dec}**")
                elif dec == "MANUAL REVIEW":
                    st.warning(f"**DECISION: {dec}**")
                else:
                    st.error(f"**DECISION: {dec}**")

                st.metric("KYC Risk Score", f"{kyc_result['risk_score']} / 100", delta=kyc_result['risk_level'], delta_color="inverse")
                b = kyc_result["breakdown"]
                st.progress(float(b["antispoof_score"]), text=f"Anti-Spoof Score: {b['antispoof_score']*100:.1f}%")
                st.progress(float(b["liveness_score"]), text=f"Liveness Score: {b['liveness_score']*100:.1f}%")
                st.progress(float(ilm_metrics.get("temporal_score", 0.5)), text=f"Temporal Score: {ilm_metrics.get('temporal_score', 0.5)*100:.1f}%")
                st.progress(float(b["face_match_score"]), text=f"Face Match Score: {b['face_match_score']*100:.1f}%")

        cap.release()

elif mode == "Sample Images Evaluation":
    col1.subheader("🖼️ Sample Image Evaluation")
    sample_name = col1.selectbox("Select Sample Image", ["image_T1.jpg (Real Face)", "image_F1.jpg (Fake/Spoof)", "image_F2.jpg (Fake/Spoof)"])
    fname = sample_name.split(" ")[0]
    sample_path = os.path.join("./images/sample/", fname)

    if os.path.exists(sample_path):
        img = cv2.imread(sample_path)
        ilm_metrics = predictor.predict_sequence([img, img])
        kyc_result = kyc_engine.evaluate_kyc(
            antispoof_score=ilm_metrics["ilm_real_prob"],
            liveness_score=ilm_metrics["liveness_score"],
            id_face_img=id_face_img,
            live_face_img=img
        )
        annotated = draw_kyc_dashboard(img.copy(), kyc_result, ilm_metrics, bbox=ilm_metrics["bbox"], fps=30.0)
        rgb_img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        col1.image(rgb_img, channels="RGB", use_container_width=True)

        with col2.container():
            st.markdown("### 📊 Verification Breakdown")
            dec = kyc_result["decision"]
            if dec == "APPROVE":
                st.success(f"**DECISION: {dec}**")
            elif dec == "MANUAL REVIEW":
                st.warning(f"**DECISION: {dec}**")
            else:
                st.error(f"**DECISION: {dec}**")

            st.json(kyc_result)
