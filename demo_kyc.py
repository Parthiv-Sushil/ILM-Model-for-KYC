# -*- coding: utf-8 -*-
"""
Interactive KYC Verification Demo with Live Video Stream & Webcam Support.
Processes webcam video feed, video files, or sample images, displaying real-time HUD overlays with
Face Detection, Real/Spoof Probability, Liveness Score, Temporal Score, Face Match Score,
KYC Risk Score, and Final Decision Badge (APPROVE / MANUAL REVIEW / REJECT).
"""

import os
import argparse
import time
import cv2
import numpy as np

from src.ilm_model.ilm_predictor import ILMPredictor
from src.kyc_risk.kyc_engine import KYCRiskEngine


def draw_kyc_dashboard(frame, kyc_result, ilm_metrics, bbox=None, fps=0.0, challenge_prompt=None):
    """
    Draws rich, professional HUD overlay on video frame.
    """
    h, w, _ = frame.shape
    overlay = frame.copy()

    risk_score = kyc_result["risk_score"]
    risk_level = kyc_result["risk_level"]
    decision = kyc_result["decision"]
    breakdown = kyc_result["breakdown"]

    # Color Palette according to decision:
    if decision == "APPROVE":
        badge_color = (0, 200, 0)
        box_color = (0, 220, 0)
    elif decision == "MANUAL REVIEW":
        badge_color = (0, 190, 255)
        box_color = (0, 190, 255)
    else:
        badge_color = (0, 0, 220)
        box_color = (0, 0, 220)

    # 1. Bounding Box around Face
    if bbox is not None and len(bbox) == 4:
        bx, by, bw, bh = bbox
        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), box_color, 2)
        label_text = f"{decision} ({risk_score:.1f})"
        cv2.rectangle(overlay, (bx, max(0, by - 25)), (bx + bw, max(0, by)), box_color, -1)
        cv2.putText(overlay, label_text, (bx + 5, max(12, by - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 2. Header Banner
    banner_height = 45
    cv2.rectangle(overlay, (0, 0), (w, banner_height), (20, 20, 20), -1)
    cv2.putText(overlay, "DIGITAL KYC VERIFICATION - LIVE VIDEO / ILM MODEL",
                (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(overlay, f"FPS: {fps:.1f}", (w - 110, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # 3. Challenge Prompt Banner (if active)
    if challenge_prompt:
        cv2.rectangle(overlay, (0, h - 35), (w, h), (40, 40, 40), -1)
        cv2.putText(overlay, f"CHALLENGE: {challenge_prompt}", (15, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # 4. Sidebar Dashboard Panel
    panel_w = 320
    panel_x = w - panel_w - 10
    panel_y = banner_height + 10
    panel_h = 240

    # Semi-transparent background
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (15, 15, 15), -1)
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (80, 80, 80), 1)

    # Dashboard Metrics List
    metrics_display = [
        ("Face Detected", "YES" if bbox is not None else "NO", (255, 255, 255)),
        ("Anti-Spoof Prob", f"{breakdown['antispoof_score']*100:.1f}%", (255, 255, 255)),
        ("Liveness Score", f"{breakdown['liveness_score']*100:.1f}%", (255, 255, 255)),
        ("Temporal Score", f"{ilm_metrics.get('temporal_score', 0.5)*100:.1f}%", (255, 255, 255)),
        ("Face Match Score", f"{breakdown['face_match_score']*100:.1f}%", (255, 255, 255)),
        ("KYC Risk Score", f"{risk_score:.1f} / 100 ({risk_level})", (0, 255, 255) if risk_level == "LOW" else (0, 165, 255)),
    ]

    curr_y = panel_y + 25
    for title, val_str, val_color in metrics_display:
        cv2.putText(overlay, f"{title}:", (panel_x + 15, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(overlay, val_str, (panel_x + 175, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, val_color, 1)
        curr_y += 26

    # Decision Badge Box
    cv2.rectangle(overlay, (panel_x + 15, curr_y + 5), (panel_x + panel_w - 15, curr_y + 45), badge_color, -1)
    badge_text = f"DECISION: {decision}"
    text_size = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    text_x = panel_x + 15 + (panel_w - 30 - text_size[0]) // 2
    cv2.putText(overlay, badge_text, (text_x, curr_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Blend overlay with original frame
    alpha = 0.88
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


def run_kyc_demo(video_path=None, webcam=False, webcam_id=0, id_image_path=None, device_id=0, max_frames=300, save_output=None):
    print("=" * 70)
    print("             STARTING LIVE VIDEO / WEBCAM KYC VERIFICATION            ")
    print("=" * 70)

    ilm_predictor = ILMPredictor(device_id=device_id)
    kyc_engine = KYCRiskEngine()

    id_face_img = cv2.imread(id_image_path) if id_image_path and os.path.exists(id_image_path) else None

    # Source setup: video file, webcam, or fallback sample loop
    cap = None
    if video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        print(f"Reading input video file: {video_path}")
    elif webcam:
        print(f"Attempting to open webcam device index {webcam_id}...")
        cap = cv2.VideoCapture(webcam_id)
        if not cap.isOpened():
            print(f"Warning: Webcam device {webcam_id} could not be opened. Falling back to sample image simulation.")
            cap = None
        else:
            print("Webcam successfully initialized!")

    frame_buffer = []
    buffer_size = 5
    writer = None
    frame_count = 0

    if cap is not None:
        fps_start = time.time()
        print("Live video stream active. Press 'q' in video window to exit.")

        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame_buffer.append(frame.copy())
            if len(frame_buffer) > buffer_size:
                frame_buffer.pop(0)

            # Predict sequence across sliding window frame buffer
            ilm_metrics = ilm_predictor.predict_sequence(frame_buffer)
            kyc_result = kyc_engine.evaluate_kyc(
                antispoof_score=ilm_metrics["ilm_real_prob"],
                liveness_score=ilm_metrics["liveness_score"],
                id_face_img=id_face_img,
                live_face_img=frame
            )

            elapsed_fps = frame_count / max(time.time() - fps_start, 1e-5)
            
            # Interactive Challenge prompt cycle
            if frame_count % 90 < 45:
                challenge = "Blink or tilt head slightly to verify liveness"
            else:
                challenge = "Hold steady facing camera"

            annotated_frame = draw_kyc_dashboard(
                frame, kyc_result, ilm_metrics, bbox=ilm_metrics["bbox"], fps=elapsed_fps, challenge_prompt=challenge
            )

            if save_output:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(save_output, fourcc, 15.0, (frame.shape[1], frame.shape[0]))
                writer.write(annotated_frame)

            # Display GUI frame window
            try:
                window_title = "KYC Live Video Verification - ILM Model"
                if frame_count == 1:
                    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_title, 960, 540)

                cv2.imshow(window_title, annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC key
                    print("User requested exit.")
                    break
            except Exception as e:
                print(f"Window display error: {e}")

        cap.release()
        if writer:
            writer.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
    else:
        # Fallback simulation loop using repository sample images
        print("Running sample image simulation loop...")
        samples = ["image_T1.jpg", "image_F1.jpg", "image_F2.jpg"]
        for sample_name in samples:
            sample_path = os.path.join("./images/sample/", sample_name)
            if os.path.exists(sample_path):
                img = cv2.imread(sample_path)
                ilm_metrics = ilm_predictor.predict_sequence([img, img])
                kyc_result = kyc_engine.evaluate_kyc(
                    antispoof_score=ilm_metrics["ilm_real_prob"],
                    liveness_score=ilm_metrics["liveness_score"],
                    live_face_img=img
                )
                annotated = draw_kyc_dashboard(img.copy(), kyc_result, ilm_metrics, bbox=ilm_metrics["bbox"], fps=30.0)
                out_name = f"./images/sample/{os.path.splitext(sample_name)[0]}_kyc_demo.jpg"
                cv2.imwrite(out_name, annotated)
                print(f"Sample demo output saved: {out_name}")
                print(f"[{sample_name}] -> Decision: {kyc_result['decision']}, Risk Score: {kyc_result['risk_score']}")

    print("\nKYC Live Demo completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KYC Live Video / Webcam Verification Demo")
    parser.add_argument("--video_path", type=str, default=None, help="Path to input video file")
    parser.add_argument("--webcam", action="store_true", help="Enable live webcam capture stream")
    parser.add_argument("--webcam_id", type=int, default=0, help="Webcam device camera ID index (default 0)")
    parser.add_argument("--id_image_path", type=str, default=None, help="Path to reference ID card photo")
    parser.add_argument("--device_id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--save_output", type=str, default=None, help="Path to save output video (.mp4)")
    args = parser.parse_args()

    run_kyc_demo(
        video_path=args.video_path,
        webcam=args.webcam,
        webcam_id=args.webcam_id,
        id_image_path=args.id_image_path,
        device_id=args.device_id,
        save_output=args.save_output
    )
