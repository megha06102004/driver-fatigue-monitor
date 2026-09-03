"""
Desktop Runner for Driver Fatigue & Drowsiness Monitoring System.
Ultra-low latency, real-time HUD with adaptive baseline calibration.
"""

import argparse
import sys
import time
import cv2

from src.detector import FatigueDetector
from src.alert import draw_hud


def parse_args():
    parser = argparse.ArgumentParser(description="DriverGuard AI - Desktop Runner")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Target capture width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Target capture height (default: 720)")
    parser.add_argument("--calib-time", type=float, default=3.0, help="Calibration duration in seconds (default: 3.0)")
    parser.add_argument("--mute", action="store_true", help="Start with audio alerts muted")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 65)
    print("  DRIVER GUARD AI - REAL-TIME DRIVER MONITORING SYSTEM")
    print("=" * 65)
    print(f"Initializing camera device index {args.camera}...")

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f"[ERROR] Failed to access camera index {args.camera}.")
        print("Please check your webcam connection or verify camera permissions.")
        return 1

    detector = FatigueDetector(
        calibration_duration_sec=args.calib_time,
        enable_audio=not args.mute
    )

    print("\nSystem Controls:")
    print("  [r] : Recalibrate personal baseline geometry")
    print("  [m] : Toggle audio alert mute / unmute")
    print("  [s] : Save current HUD screenshot")
    print("  [q] : Quit application\n")
    print("[INFO] Starting video stream... Please look straight ahead for initial calibration.")

    prev_time = time.time()
    fps = 30.0

    window_name = "DriverGuard AI - Driver Monitoring System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.width, args.height)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Blank frame received from webcam. Re-trying...")
                time.sleep(0.05)
                continue

            # Flip horizontally for natural mirror view
            frame = cv2.flip(frame, 1)

            # Compute FPS with exponential moving average
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            # Process frame through FatigueDetector
            annotated_frame, telemetry = detector.process_frame(frame)

            # Draw high-tech HUD overlay
            hud_frame = draw_hud(annotated_frame, telemetry, fps)

            # Render frame
            cv2.imshow(window_name, hud_frame)

            # Keyboard shortcuts
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # 'q' or ESC
                print("[INFO] Exiting...")
                break
            elif key == ord('r'):
                print("[INFO] Recalibrating baseline face geometry...")
                detector.reset_calibration()
            elif key == ord('m'):
                detector.audio.enabled = not detector.audio.enabled
                state_str = "MUTED" if not detector.audio.enabled else "ACTIVE"
                print(f"[INFO] Audio alert is now {state_str}")
            elif key == ord('s'):
                filename = f"drowsiness_capture_{int(time.time())}.png"
                cv2.imwrite(filename, hud_frame)
                print(f"[INFO] Screenshot saved to {filename}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released. Goodbye!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
