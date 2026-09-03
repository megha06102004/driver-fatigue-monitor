"""
Unit and Integration Tests for Metrics, Landmark Processing, and Audio Alerting.
"""

import unittest
import numpy as np
import cv2

from src.metrics import (
    calculate_ear,
    calculate_mar,
    calculate_head_pose,
    euclidean_distance
)
from src.landmarks import GENERIC_FACE_3D_MODEL, HEAD_POSE_INDICES
from src.alert import AudioManager


class TestDriverGuardMetrics(unittest.TestCase):

    def test_euclidean_distance(self):
        p1 = np.array([0.0, 0.0])
        p2 = np.array([3.0, 4.0])
        dist = euclidean_distance(p1, p2)
        self.assertAlmostEqual(dist, 5.0)

    def test_ear_open_vs_closed(self):
        # Synthetic open eye coordinates
        # [p1(corner1), p2(top1), p3(top2), p4(corner2), p5(bot2), p6(bot1)]
        open_eye = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 10.0, 0.0],
            [20.0, 10.0, 0.0],
            [30.0, 0.0, 0.0],
            [20.0, -10.0, 0.0],
            [10.0, -10.0, 0.0],
        ])
        open_ear = calculate_ear(open_eye)
        self.assertGreater(open_ear, 0.5)

        # Synthetic closed eye coordinates (vertical points collapsed)
        closed_eye = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 1.0, 0.0],
            [20.0, 1.0, 0.0],
            [30.0, 0.0, 0.0],
            [20.0, -1.0, 0.0],
            [10.0, -1.0, 0.0],
        ])
        closed_ear = calculate_ear(closed_eye)
        self.assertLess(closed_ear, 0.15)
        self.assertLess(closed_ear, open_ear)

    def test_head_pose_estimation(self):
        # Create a mock array of 468 landmarks matching the 3D model projected onto image plane
        landmarks = np.zeros((468, 3), dtype=np.float64)
        img_w, img_h = 640, 480
        center_x, center_y = 320.0, 240.0

        # Place the 6 head pose points near screen center
        landmarks[1] = [center_x, center_y, 0.0]                  # Nose
        landmarks[152] = [center_x, center_y + 80.0, 0.0]         # Chin
        landmarks[33] = [center_x - 50.0, center_y - 40.0, 0.0]   # Right eye
        landmarks[263] = [center_x + 50.0, center_y - 40.0, 0.0]  # Left eye
        landmarks[61] = [center_x - 30.0, center_y + 40.0, 0.0]   # Right mouth
        landmarks[291] = [center_x + 30.0, center_y + 40.0, 0.0]  # Left mouth

        pitch, yaw, roll, rvec, tvec = calculate_head_pose(landmarks, img_w, img_h)
        self.assertIsInstance(pitch, float)
        self.assertIsInstance(yaw, float)
        self.assertIsInstance(roll, float)
        self.assertEqual(rvec.shape, (3, 1))
        self.assertEqual(tvec.shape, (3, 1))

    def test_audio_manager_non_blocking(self):
        audio = AudioManager(enabled=False)  # Disabled so no loud beep in test
        audio.trigger_alarm(severity="warning")
        self.assertFalse(audio.is_playing)


if __name__ == "__main__":
    unittest.main()
