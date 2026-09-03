"""
MediaPipe FaceMesh Landmark Definitions and Utility Functions.
"""

from typing import List, Tuple, Dict
import numpy as np

# MediaPipe FaceMesh indices for Eye Aspect Ratio (EAR)
# 6 points per eye: [p1(corner1), p2(top1), p3(top2), p4(corner2), p5(bottom2), p6(bottom1)]
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# Mouth indices for Mouth Aspect Ratio (MAR) - inner lip contour for sensitive yawn detection
# Corner points: 78, 308
# Vertical pairs: (81, 178), (13, 14), (311, 402)
MOUTH_INNER_INDICES = {
    "corners": (78, 308),
    "vertical_pairs": [(81, 178), (13, 14), (311, 402)]
}

# 2D/3D landmarks used for Head Pose Estimation (Perspective-n-Point)
# 1: Nose tip
# 152: Chin
# 33: Right eye outer corner (in image left)
# 263: Left eye outer corner (in image right)
# 61: Right mouth corner
# 291: Left mouth corner
HEAD_POSE_INDICES = [1, 152, 33, 263, 61, 291]

# 3D generic facial model coordinates (in millimeters, centered at nose tip)
GENERIC_FACE_3D_MODEL = np.array([
    [0.0, 0.0, 0.0],          # Nose tip (1)
    [0.0, -330.0, -65.0],      # Chin (152)
    [-225.0, 170.0, -135.0],   # Right eye outer corner (33)
    [225.0, 170.0, -135.0],    # Left eye outer corner (263)
    [-150.0, -150.0, -125.0],  # Right mouth corner (61)
    [150.0, -150.0, -125.0]    # Left mouth corner (291)
], dtype=np.float64)


def extract_landmarks_pixel(face_landmarks, img_w: int, img_h: int) -> np.ndarray:
    """
    Extract all 468/478 landmarks and convert normalized coordinates to pixel coordinates.
    Returns an (N, 3) numpy array [x, y, z] in pixels.
    """
    coords = np.zeros((len(face_landmarks.landmark), 3), dtype=np.float64)
    for i, lm in enumerate(face_landmarks.landmark):
        coords[i] = [lm.x * img_w, lm.y * img_h, lm.z * img_w]
    return coords
