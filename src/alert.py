"""
Audio and Visual Alert Engine for Driver Fatigue Monitoring.
Provides non-blocking audio alerts and a polished, modern HUD overlay.
"""

import sys
import threading
import time
from typing import Optional, Dict, Any
import cv2
import numpy as np


class AudioManager:
    """Non-blocking asynchronous audio alert manager using native Windows winsound."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.is_playing = False
        self._lock = threading.Lock()
        self._last_alert_time = 0.0

    def trigger_alarm(self, severity: str = "critical"):
        """
        Triggers an audio beep pattern asynchronously on a worker thread.
        Severity:
          - "critical": high-pitch rapid double beep for severe drowsiness / head drop
          - "warning": moderate beep for yawn / distraction
        """
        if not self.enabled:
            return

        now = time.time()
        # Rate limit alert trigger to prevent overlapping screeching
        if now - self._last_alert_time < 0.6:
            return

        with self._lock:
            if self.is_playing:
                return
            self.is_playing = True
            self._last_alert_time = now

        threading.Thread(target=self._play_worker, args=(severity,), daemon=True).start()

    def _play_worker(self, severity: str):
        try:
            if sys.platform == "win32":
                import winsound
                if severity == "critical":
                    # Urgent double-beep: 2000Hz for 180ms x 2
                    winsound.Beep(2200, 180)
                    time.sleep(0.08)
                    winsound.Beep(2200, 220)
                else:
                    # Caution beep: 1500Hz for 200ms
                    winsound.Beep(1500, 200)
            else:
                # Fallback for Linux/macOS
                print("\a", end="", flush=True)
        except Exception:
            pass
        finally:
            with self._lock:
                self.is_playing = False


def draw_hud(
    frame: np.ndarray,
    data: Dict[str, Any],
    fps: float
) -> np.ndarray:
    """
    Renders a high-tech, modern Driver Monitoring HUD directly onto the OpenCV frame.
    Displays:
      - Driver Status Pill (NORMAL, DROWSY, YAWN, DISTRACTED, CALIBRATING)
      - Dynamic metric bars (EAR, MAR, Head Pitch, Head Yaw)
      - Session statistics (Events, Blinks, Yawns)
      - Live FPS counter
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    status = data.get("status", "NORMAL")
    is_calibrating = data.get("is_calibrating", False)
    calib_progress = data.get("calibration_progress", 1.0)
    ear = data.get("ear", 0.0)
    mar = data.get("mar", 0.0)
    pitch = data.get("pitch", 0.0)
    yaw = data.get("yaw", 0.0)
    ear_thresh = data.get("ear_threshold", 0.22)
    mar_thresh = data.get("mar_threshold", 0.50)

    # Status color theme (BGR)
    colors = {
        "NORMAL": (60, 220, 80),        # Vibrant Green
        "CALIBRATING": (255, 190, 0),    # Cyan/Amber
        "BLINK": (200, 200, 200),       # Light Grey
        "YAWN": (30, 144, 255),         # Dodger Blue / Orange
        "DISTRACTED": (255, 69, 0),     # Red-Orange
        "HEAD_DROP": (0, 0, 255),       # Crimson Red
        "DROWSY": (0, 0, 255)           # Crimson Red
    }
    theme_color = colors.get(status, (255, 255, 255))

    # 1. Top Status Banner Card (Semi-transparent dark header)
    header_h = 75
    cv2.rectangle(overlay, (0, 0), (w, header_h), (20, 22, 28), -1)

    # Status Pill
    pill_w = 200
    pill_x1 = 20
    pill_y1 = 15
    pill_x2 = pill_x1 + pill_w
    pill_y2 = pill_y1 + 45
    cv2.rectangle(overlay, (pill_x1, pill_y1), (pill_x2, pill_y2), theme_color, -1)

    # Blend header
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Status Text inside Pill
    status_display = "CALIBRATING" if is_calibrating else status
    text_size = cv2.getTextSize(status_display, cv2.FONT_HERSHEY_DUPLEX, 0.65, 2)[0]
    tx = pill_x1 + (pill_w - text_size[0]) // 2
    ty = pill_y1 + (45 + text_size[1]) // 2
    # Contrast text: dark for green/cyan, white for red/blue
    text_color = (15, 15, 15) if status in ("NORMAL", "CALIBRATING") else (255, 255, 255)
    cv2.putText(frame, status_display, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 0.65, text_color, 2)

    # Title & FPS
    cv2.putText(frame, "DRIVER GUARD AI", (pill_x2 + 25, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (pill_x2 + 25, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    # Session Stats (Right of Header)
    stats_str = f"Events: {data.get('alert_count', 0)}  |  Yawns: {data.get('yawn_count', 0)}"
    stats_size = cv2.getTextSize(stats_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.putText(frame, stats_str, (w - stats_size[0] - 25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    # 2. Calibration Progress Bar (if active)
    if is_calibrating:
        bar_w = int(w * calib_progress)
        cv2.rectangle(frame, (0, header_h - 4), (bar_w, header_h), (255, 200, 0), -1)
        cv2.putText(frame, f"Calibrating baseline face geometry: {int(calib_progress * 100)}%",
                    (20, header_h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 100), 2)

    # 3. Bottom Telemetry Metric Panel (Sidebar / HUD)
    hud_x = 20
    hud_y = h - 130
    panel_w = 320
    panel_h = 115

    hud_overlay = frame.copy()
    cv2.rectangle(hud_overlay, (hud_x - 10, hud_y - 15), (hud_x + panel_w, hud_y + panel_h), (15, 18, 22), -1)
    cv2.addWeighted(hud_overlay, 0.70, frame, 0.30, 0, frame)
    cv2.rectangle(frame, (hud_x - 10, hud_y - 15), (hud_x + panel_w, hud_y + panel_h), (60, 65, 75), 1)

    # Draw Metric Bar Helper
    def draw_bar(label, value, threshold, max_val, y_pos, color, warning_condition):
        cv2.putText(frame, f"{label}: {value:.2f}", (hud_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        bar_start_x = hud_x + 95
        bar_max_w = 190
        fill_w = int(np.clip(value / max_val, 0.0, 1.0) * bar_max_w)
        thresh_x = int(np.clip(threshold / max_val, 0.0, 1.0) * bar_max_w) + bar_start_x

        # Background track
        cv2.rectangle(frame, (bar_start_x, y_pos - 10), (bar_start_x + bar_max_w, y_pos), (50, 55, 65), -1)
        # Active fill
        active_color = (0, 0, 255) if warning_condition else color
        cv2.rectangle(frame, (bar_start_x, y_pos - 10), (bar_start_x + fill_w, y_pos), active_color, -1)
        # Threshold marker
        cv2.line(frame, (thresh_x, y_pos - 13), (thresh_x, y_pos + 3), (255, 255, 255), 1)

    # Metric Bars
    draw_bar("EAR", ear, ear_thresh, 0.45, hud_y + 8, (60, 220, 80), ear < ear_thresh)
    draw_bar("MAR", mar, mar_thresh, 0.80, hud_y + 35, (30, 144, 255), mar > mar_thresh)
    draw_bar("Pitch", pitch, 20.0, 45.0, hud_y + 62, (200, 180, 50), pitch > 20.0 or pitch < -20.0)
    draw_bar("Yaw", abs(yaw), 25.0, 50.0, hud_y + 89, (200, 120, 220), abs(yaw) > 25.0)

    # 4. Flashing Full-Screen Red Warning Border if Drowsy / Nodding
    if status in ("DROWSY", "HEAD_DROP"):
        flash = int(time.time() * 5) % 2 == 0
        if flash:
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 8)
            warning_msg = "! WAKE UP - DROWSINESS DETECTED !" if status == "DROWSY" else "! HEAD DROP DETECTED !"
            wsize = cv2.getTextSize(warning_msg, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)[0]
            wx = (w - wsize[0]) // 2
            cv2.rectangle(frame, (wx - 15, h // 2 - 30), (wx + wsize[0] + 15, h // 2 + 15), (0, 0, 220), -1)
            cv2.putText(frame, warning_msg, (wx, h // 2), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)

    elif status == "DISTRACTED":
        flash = int(time.time() * 4) % 2 == 0
        if flash:
            cv2.rectangle(frame, (0, 0), (w, h), (255, 69, 0), 6)
            d_msg = "! FOCUS ON ROAD - DISTRACTION DETECTED !"
            dsize = cv2.getTextSize(d_msg, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)[0]
            dx = (w - dsize[0]) // 2
            cv2.rectangle(frame, (dx - 12, h // 2 - 25), (dx + dsize[0] + 12, h // 2 + 12), (255, 69, 0), -1)
            cv2.putText(frame, d_msg, (dx, h // 2), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

    return frame
