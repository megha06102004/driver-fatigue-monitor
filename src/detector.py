"""
Driver Fatigue & Distraction Detector.
Integrates MediaPipe FaceMesh, metrics calculations, adaptive calibration, and state machine.
"""

import time
from typing import Dict, Any, Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np

from .landmarks import (
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    extract_landmarks_pixel,
)
from .metrics import (
    calculate_ear,
    calculate_mar,
    calculate_head_pose,
    draw_pose_axes,
)
from .alert import AudioManager
from .cnn_classifier import EyeCNNClassifier, crop_eye_region


class FatigueDetector:
    """
    State-of-the-art Fatigue and Distraction Detector.
    Features:
      - MediaPipe FaceMesh landmark extraction (468 points)
      - Self-calibrating eye/mouth baseline
      - Dual-eye EAR and MAR tracking
      - 3D Head Pose tracking for nodding & distraction
      - Temporal state machine filtering out natural blinks
    """

    def __init__(
        self,
        calibration_duration_sec: float = 3.0,
        eye_close_threshold_ratio: float = 0.75,  # 75% of calibrated baseline EAR
        yawn_threshold_ratio: float = 1.65,       # 165% of calibrated baseline MAR
        min_ear_threshold: float = 0.21,
        min_mar_threshold: float = 0.45,
        drowsy_time_sec: float = 1.2,
        yawn_time_sec: float = 1.5,
        distraction_time_sec: float = 2.0,
        head_nod_pitch_threshold: float = 20.0,
        head_yaw_distraction_threshold: float = 25.0,
        enable_audio: bool = True
    ):
        self.calibration_duration_sec = calibration_duration_sec
        self.eye_close_threshold_ratio = eye_close_threshold_ratio
        self.yawn_threshold_ratio = yawn_threshold_ratio
        self.min_ear_threshold = min_ear_threshold
        self.min_mar_threshold = min_mar_threshold

        self.drowsy_time_sec = drowsy_time_sec
        self.yawn_time_sec = yawn_time_sec
        self.distraction_time_sec = distraction_time_sec
        self.head_nod_pitch_threshold = head_nod_pitch_threshold
        self.head_yaw_distraction_threshold = head_yaw_distraction_threshold

        # MediaPipe FaceMesh setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Audio Manager
        self.audio = AudioManager(enabled=enable_audio)

        # CNN Eye State Classifier
        self.cnn = EyeCNNClassifier()

        # Calibration state
        self.is_calibrating = True
        self.calib_start_time = None
        self.calib_ear_samples = []
        self.calib_mar_samples = []

        # Default fallback thresholds
        self.ear_threshold = min_ear_threshold
        self.mar_threshold = min_mar_threshold

        # Temporal counters & timestamps
        self.eye_closed_start_time: Optional[float] = None
        self.yawn_start_time: Optional[float] = None
        self.distraction_start_time: Optional[float] = None
        self.last_state = "NORMAL"

        # Session metrics
        self.total_blinks = 0
        self.total_yawns = 0
        self.total_alerts = 0
        self._in_blink = False
        self._in_yawn = False

    def reset_calibration(self):
        """Resets the personalized baseline calibration."""
        self.is_calibrating = True
        self.calib_start_time = None
        self.calib_ear_samples.clear()
        self.calib_mar_samples.clear()

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Processes a single BGR camera frame and returns:
          - Annotated frame (with head pose axes and landmark visualizers)
          - Telemetry dictionary for HUD and dashboard
        """
        img_h, img_w = frame.shape[:2]
        now = time.time()

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True

        data = {
            "face_detected": False,
            "status": "NO_FACE",
            "ear": 0.0,
            "left_ear": 0.0,
            "right_ear": 0.0,
            "mar": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "roll": 0.0,
            "is_calibrating": self.is_calibrating,
            "calibration_progress": 0.0,
            "ear_threshold": self.ear_threshold,
            "mar_threshold": self.mar_threshold,
            "blink_count": self.total_blinks,
            "yawn_count": self.total_yawns,
            "alert_count": self.total_alerts,
        }

        if not results.multi_face_landmarks:
            # Face lost - reset transient timers
            self.eye_closed_start_time = None
            self.yawn_start_time = None
            self.distraction_start_time = None
            return frame, data

        face_landmarks = results.multi_face_landmarks[0]
        data["face_detected"] = True

        # Extract 2D/3D pixel landmarks
        landmarks = extract_landmarks_pixel(face_landmarks, img_w, img_h)

        # 1. Compute EAR for both eyes
        left_eye_pts = landmarks[LEFT_EYE_INDICES]
        right_eye_pts = landmarks[RIGHT_EYE_INDICES]

        left_ear = calculate_ear(left_eye_pts)
        right_ear = calculate_ear(right_eye_pts)
        avg_ear = (left_ear + right_ear) / 2.0

        # 2. CNN-Based Eye-State Classification
        left_crop = crop_eye_region(frame, left_eye_pts[:, :2])
        right_crop = crop_eye_region(frame, right_eye_pts[:, :2])
        left_cnn_state, left_cnn_prob = self.cnn.predict_eye(left_crop)
        right_cnn_state, right_cnn_prob = self.cnn.predict_eye(right_crop)
        cnn_eyes_closed = (left_cnn_state == "Closed" and right_cnn_state == "Closed")

        # 3. Compute MAR
        mar = calculate_mar(landmarks)

        # 4. Compute 3D Head Pose
        pitch, yaw, roll, rvec, tvec = calculate_head_pose(landmarks, img_w, img_h)

        # Render 3D pose coordinate axes on nose
        frame = draw_pose_axes(frame, rvec, tvec, img_w, img_h, axis_length=70.0)

        # Draw subtle eye contours
        for pt in np.vstack([left_eye_pts[:, :2], right_eye_pts[:, :2]]).astype(int):
            cv2.circle(frame, tuple(pt), 1, (0, 255, 180), -1)

        data.update({
            "ear": avg_ear,
            "left_ear": left_ear,
            "right_ear": right_ear,
            "cnn_left_state": left_cnn_state,
            "cnn_right_state": right_cnn_state,
            "cnn_left_prob": left_cnn_prob,
            "cnn_right_prob": right_cnn_prob,
            "cnn_eyes_closed": cnn_eyes_closed,
            "mar": mar,
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
        })

        # 4. Handle Calibration Phase
        if self.is_calibrating:
            if self.calib_start_time is None:
                self.calib_start_time = now

            elapsed = now - self.calib_start_time
            progress = min(1.0, elapsed / self.calibration_duration_sec)
            data["calibration_progress"] = progress

            # Collect samples when looking relatively forward
            if abs(yaw) < 20.0 and abs(pitch) < 20.0:
                self.calib_ear_samples.append(avg_ear)
                self.calib_mar_samples.append(mar)

            if elapsed >= self.calibration_duration_sec and len(self.calib_ear_samples) > 20:
                # Compute calibrated baseline
                baseline_ear = float(np.median(self.calib_ear_samples))
                baseline_mar = float(np.median(self.calib_mar_samples))

                self.ear_threshold = max(self.min_ear_threshold, baseline_ear * self.eye_close_threshold_ratio)
                self.mar_threshold = max(self.min_mar_threshold, baseline_mar * self.yawn_threshold_ratio)

                self.is_calibrating = False
                data["is_calibrating"] = False
                data["ear_threshold"] = self.ear_threshold
                data["mar_threshold"] = self.mar_threshold

            data["status"] = "CALIBRATING"
            return frame, data

        # 5. Fatigue & Distraction State Machine
        status = "NORMAL"

        # A. Eye Closure / Drowsiness Check (Hybrid: Geometric EAR + CNN Video Frame Classifier)
        is_closed = (avg_ear < self.ear_threshold) or cnn_eyes_closed
        if is_closed:
            if self.eye_closed_start_time is None:
                self.eye_closed_start_time = now

            duration = now - self.eye_closed_start_time
            if duration >= self.drowsy_time_sec:
                status = "DROWSY"
                self.audio.trigger_alarm(severity="critical")
                if self.last_state != "DROWSY":
                    self.total_alerts += 1
            elif duration > 0.15:
                status = "BLINK"
                if not self._in_blink:
                    self.total_blinks += 1
                    self._in_blink = True
        else:
            self.eye_closed_start_time = None
            self._in_blink = False

        # B. Head Nodding Check (Micro-sleep indicator)
        # Pitch > 18 deg forward or sudden downward nod
        if pitch > self.head_nod_pitch_threshold:
            status = "HEAD_DROP"
            self.audio.trigger_alarm(severity="critical")
            if self.last_state != "HEAD_DROP":
                self.total_alerts += 1

        # C. Yawn Check
        if mar > self.mar_threshold:
            if self.yawn_start_time is None:
                self.yawn_start_time = now

            duration = now - self.yawn_start_time
            if duration >= self.yawn_time_sec:
                if status == "NORMAL":
                    status = "YAWN"
                self.audio.trigger_alarm(severity="warning")
                if not self._in_yawn:
                    self.total_yawns += 1
                    self.total_alerts += 1
                    self._in_yawn = True
        else:
            self.yawn_start_time = None
            self._in_yawn = False

        # D. Head Distraction Check (Driver looking away left/right)
        if abs(yaw) > self.head_yaw_distraction_threshold:
            if self.distraction_start_time is None:
                self.distraction_start_time = now

            duration = now - self.distraction_start_time
            if duration >= self.distraction_time_sec:
                if status == "NORMAL":
                    status = "DISTRACTED"
                self.audio.trigger_alarm(severity="warning")
                if self.last_state != "DISTRACTED":
                    self.total_alerts += 1
        else:
            self.distraction_start_time = None

        self.last_state = status
        data["status"] = status
        data["blink_count"] = self.total_blinks
        data["yawn_count"] = self.total_yawns
        data["alert_count"] = self.total_alerts

        return frame, data
