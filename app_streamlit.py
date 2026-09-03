"""
Interactive Streamlit Web Dashboard for Driver Fatigue & Drowsiness Monitoring.
Ideal for portfolio showcases and live demonstrations.
"""

import time
import cv2
import numpy as np
import streamlit as st

from src.detector import FatigueDetector
from src.alert import draw_hud

# Page configuration
st.set_page_config(
    page_title="DriverGuard AI - Driver Monitoring System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #888;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1a1e24;
        border: 1px solid #2e3642;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.markdown('<div class="main-header">🚗 DriverGuard AI: Real-Time Fatigue Monitoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Edge Computer Vision powered by Google MediaPipe FaceMesh, 3D Head Pose, & Adaptive EAR/MAR</div>', unsafe_allow_html=True)

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ System Configuration")
        st.markdown("Fine-tune fatigue sensitivity parameters:")

        ear_ratio = st.slider("EAR Sensitivity (% of baseline)", min_value=60, max_value=90, value=75, step=5) / 100.0
        mar_ratio = st.slider("MAR Sensitivity (% of baseline)", min_value=120, max_value=220, value=165, step=5) / 100.0
        drowsy_time = st.slider("Drowsy Threshold (seconds)", min_value=0.8, max_value=3.0, value=1.2, step=0.1)
        distract_yaw = st.slider("Distraction Yaw Angle (°)", min_value=15, max_value=45, value=25, step=5)
        nod_pitch = st.slider("Head Nod Pitch Angle (°)", min_value=10, max_value=35, value=20, step=2)

        enable_sound = st.checkbox("Enable Native Audio Alarms", value=True)
        st.markdown("---")
        st.markdown("### 📘 System Architecture")
        st.info("""
        - **Face Mesh:** 468 3D Landmarks
        - **Eye State:** 6-point EAR
        - **Yawn Detection:** Inner-lip MAR
        - **Distraction:** Perspective-n-Point Pose
        - **Calibration:** 3-second auto baseline
        """)

    # Main columns
    col_video, col_telemetry = st.columns([2.2, 1])

    with col_telemetry:
        st.subheader("📊 Live Telemetry")
        metric_status = st.empty()
        col1, col2 = st.columns(2)
        with col1:
            ear_metric = st.empty()
            pitch_metric = st.empty()
            blink_metric = st.empty()
        with col2:
            mar_metric = st.empty()
            yaw_metric = st.empty()
            alert_metric = st.empty()

        st.markdown("---")
        ear_bar = st.progress(0, text="Eye Aspect Ratio (EAR)")
        mar_bar = st.progress(0, text="Mouth Aspect Ratio (MAR)")

    with col_video:
        video_placeholder = st.empty()
        run_camera = st.toggle("🎥 Start Camera Stream", value=False)

    if not run_camera:
        video_placeholder.info("Click 'Start Camera Stream' to activate your webcam and test driver monitoring in real time.")
        return

    # Initialize Detector
    detector = FatigueDetector(
        calibration_duration_sec=3.0,
        eye_close_threshold_ratio=ear_ratio,
        yawn_threshold_ratio=mar_ratio,
        drowsy_time_sec=drowsy_time,
        head_nod_pitch_threshold=float(nod_pitch),
        head_yaw_distraction_threshold=float(distract_yaw),
        enable_audio=enable_sound
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot access webcam (Index 0). Please ensure your camera is not being used by another application.")
        return

    prev_time = time.time()
    fps = 30.0

    try:
        while run_camera:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.03)
                continue

            frame = cv2.flip(frame, 1)

            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            # Process frame
            annotated_frame, telemetry = detector.process_frame(frame)
            hud_frame = draw_hud(annotated_frame, telemetry, fps)

            # Convert BGR to RGB for Streamlit display
            rgb_hud = cv2.cvtColor(hud_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb_hud, channels="RGB", use_container_width=True)

            # Update Telemetry Sidebar
            status = telemetry["status"]
            ear = telemetry["ear"]
            mar = telemetry["mar"]
            pitch = telemetry["pitch"]
            yaw = telemetry["yaw"]

            status_colors = {
                "NORMAL": "🟢 NORMAL (Alert)",
                "CALIBRATING": "🟡 CALIBRATING BASELINE...",
                "DROWSY": "🔴 DROWSINESS ALERT!",
                "HEAD_DROP": "🔴 HEAD NOD DETECTED!",
                "YAWN": "🟠 YAWNING DETECTED",
                "DISTRACTED": "🟣 DRIVER DISTRACTED"
            }
            metric_status.markdown(f"### Status: {status_colors.get(status, status)}")

            ear_metric.metric("EAR", f"{ear:.3f}", f"Threshold: {telemetry['ear_threshold']:.2f}")
            mar_metric.metric("MAR", f"{mar:.3f}", f"Threshold: {telemetry['mar_threshold']:.2f}")
            pitch_metric.metric("Head Pitch", f"{pitch:.1f}°")
            yaw_metric.metric("Head Yaw", f"{yaw:.1f}°")
            blink_metric.metric("Total Blinks", telemetry["blink_count"])
            alert_metric.metric("Total Alerts", telemetry["alert_count"])

            ear_bar.progress(float(np.clip(ear / 0.45, 0.0, 1.0)), text=f"Eye Closure Ratio (EAR): {ear:.2f}")
            mar_bar.progress(float(np.clip(mar / 0.80, 0.0, 1.0)), text=f"Mouth Open Ratio (MAR): {mar:.2f}")

    finally:
        cap.release()


if __name__ == "__main__":
    main()
