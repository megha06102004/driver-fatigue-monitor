"""
Mathematical calculations for EAR, MAR, and 3D Head Pose Estimation.
"""

from typing import Tuple
import cv2
import numpy as np

from .landmarks import (
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    MOUTH_INNER_INDICES,
    HEAD_POSE_INDICES,
    GENERIC_FACE_3D_MODEL,
)


def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Computes standard 2D/3D euclidean distance between two points."""
    return float(np.linalg.norm(p1 - p2))


def calculate_ear(eye_coords: np.ndarray) -> float:
    """
    Computes Eye Aspect Ratio (EAR) given 6 eye landmarks.
    Indices: [corner1, top1, top2, corner2, bot2, bot1]
    Formula: (|top1 - bot1| + |top2 - bot2|) / (2.0 * |corner1 - corner2|)
    """
    if eye_coords.shape[0] < 6:
        return 0.0

    p1, p2, p3, p4, p5, p6 = eye_coords[:6]

    v1 = euclidean_distance(p2[:2], p6[:2])
    v2 = euclidean_distance(p3[:2], p5[:2])
    h = euclidean_distance(p1[:2], p4[:2])

    if h < 1e-6:
        return 0.0

    return (v1 + v2) / (2.0 * h)


def calculate_mar(landmarks: np.ndarray) -> float:
    """
    Computes Mouth Aspect Ratio (MAR) from inner lips for sensitive yawn detection.
    """
    c1_idx, c2_idx = MOUTH_INNER_INDICES["corners"]
    c1 = landmarks[c1_idx][:2]
    c2 = landmarks[c2_idx][:2]

    horizontal = euclidean_distance(c1, c2)
    if horizontal < 1e-6:
        return 0.0

    vertical_sum = 0.0
    for top_idx, bot_idx in MOUTH_INNER_INDICES["vertical_pairs"]:
        top = landmarks[top_idx][:2]
        bot = landmarks[bot_idx][:2]
        vertical_sum += euclidean_distance(top, bot)

    num_pairs = len(MOUTH_INNER_INDICES["vertical_pairs"])
    return vertical_sum / (num_pairs * horizontal)


def calculate_head_pose(
    landmarks: np.ndarray, img_w: int, img_h: int
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Estimates 3D head pose (Pitch, Yaw, Roll) in degrees using cv2.solvePnP.
    Returns:
        pitch: degrees (nodding forward is negative/positive depending on convention, here forward = positive tilt down)
        yaw: degrees (turning left = negative, right = positive)
        roll: degrees (tilting head side to side)
        rvec: rotation vector from solvePnP
        tvec: translation vector from solvePnP
    """
    # 2D image points from landmarks
    image_points = np.array([
        landmarks[idx][:2] for idx in HEAD_POSE_INDICES
    ], dtype=np.float64)

    # Approximate camera intrinsic matrix (pinhole model assumption)
    focal_length = float(img_w)
    center = (img_w / 2.0, img_h / 2.0)
    camera_matrix = np.array([
        [focal_length, 0.0, center[0]],
        [0.0, focal_length, center[1]],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        GENERIC_FACE_3D_MODEL,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0, 0.0, 0.0, np.zeros((3, 1)), np.zeros((3, 1))

    # Convert rotation vector to 3x3 rotation matrix
    rmat, _ = cv2.Rodrigues(rvec)

    # Decompose rotation matrix into Euler angles
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    pitch, yaw, roll = angles[0], angles[1], angles[2]

    # Convert to standard degrees
    # Pitch: up/down rotation (nodding)
    # Yaw: left/right rotation
    # Roll: lateral tilt
    return float(pitch), float(yaw), float(roll), rvec, tvec


def draw_pose_axes(
    frame: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    img_w: int,
    img_h: int,
    axis_length: float = 100.0
) -> np.ndarray:
    """Projects 3D coordinate axes onto the nose to visualize head orientation."""
    focal_length = float(img_w)
    center = (img_w / 2.0, img_h / 2.0)
    camera_matrix = np.array([
        [focal_length, 0.0, center[0]],
        [0.0, focal_length, center[1]],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # 3D points representing axes (Origin, X-red, Y-green, Z-blue)
    axis_points = np.array([
        [0.0, 0.0, 0.0],
        [axis_length, 0.0, 0.0],
        [0.0, -axis_length, 0.0],
        [0.0, 0.0, -axis_length]
    ], dtype=np.float64)

    imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = imgpts.reshape(-1, 2).astype(int)

    origin = tuple(imgpts[0])
    # X Axis: Red (pitch/lateral)
    cv2.line(frame, origin, tuple(imgpts[1]), (0, 0, 255), 2)
    # Y Axis: Green (up/down)
    cv2.line(frame, origin, tuple(imgpts[2]), (0, 255, 0), 2)
    # Z Axis: Blue (pointing outward from nose)
    cv2.line(frame, origin, tuple(imgpts[3]), (255, 128, 0), 2)

    return frame
