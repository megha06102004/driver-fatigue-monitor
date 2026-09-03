# 🚗 DriverGuard AI: Real-Time Driver Fatigue & Distraction Monitoring System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-FaceMesh%20(468%20pts)-brightgreen.svg)](https://developers.google.com/mediapipe)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red.svg)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent, multi-modal Driver Monitoring System (DMS) engineered to detect drowsiness, micro-sleep, yawning, and driver distraction in real time. Powered by **Google MediaPipe FaceMesh**, geometric **Eye Aspect Ratio (EAR)**, **Mouth Aspect Ratio (MAR)**, and **3D Head Pose Estimation (`cv2.solvePnP`)**.

---

## ⚡ Key Innovations & Improvements over Legacy Approaches

| Feature | Legacy Haar Cascade + CNN | **DriverGuard AI (This Project)** |
| :--- | :--- | :--- |
| **Tracking Pipeline** | OpenCV Haar Cascades (Bounding box) | **468 3D Facial Landmarks** (MediaPipe Face Mesh) |
| **Eye Closure Robustness** | ❌ Fails when eyes are closed (drops detection) | ✅ **100% Robust** (tracks eyelid vertices directly) |
| **Head Angles & Tilts** | ❌ Drops tracking on $15^\circ$ head turn | ✅ **Full 3D Tracking** up to $60^\circ$ yaw/pitch |
| **Distraction Monitoring** | ❌ None | ✅ **3D Perspective-n-Point (Pitch, Yaw, Roll)** |
| **Micro-Sleep Nodding** | ❌ None | ✅ **Abrupt Head-Drop Detection** |
| **False Positive Handling** | ❌ Static threshold (fails on narrow eyes) | ✅ **Self-Calibrating Baseline** in first 3 seconds |
| **Alert Engine** | ❌ `playsound` (freezes video, crashes on Win) | ✅ **Non-blocking Asynchronous Audio & Modern HUD** |
| **User Interface** | Raw OpenCV window | **Dual Interface**: Desktop HUD + Streamlit Web App |

---

## 📐 Mathematical Formulation

### 1. Eye Aspect Ratio (EAR)
Eye state is classified using the Euclidean distances between 6 canonical eyelid landmarks:

$$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \cdot \|p_1 - p_4\|}$$

Where $p_1, p_4$ are the eye corners, and $(p_2, p_6)$, $(p_3, p_5)$ are vertical eyelid landmark pairs.
* An open eye typically exhibits $\text{EAR} \approx 0.28 - 0.38$.
* A closed eye drops to $\text{EAR} < 0.18$.
* Sustained low EAR for $>1.2\text{ s}$ triggers a **DROWSY** critical alarm (distinguishing natural blinks from micro-sleep).

### 2. Mouth Aspect Ratio (MAR)
Yawn detection tracks the inner contour of the lips to measure vertical mouth opening relative to mouth width:

$$\text{MAR} = \frac{\sum_{i=1}^{3} \|top_i - bot_i\|}{3 \cdot \|corner_1 - corner_2\|}$$

* Normal talking/rest: $\text{MAR} < 0.35$.
* Yawn state: $\text{MAR} > 0.50$ sustained for $>1.5\text{ s}$.

### 3. 3D Head Pose Estimation (Pitch, Yaw, Roll)
By mapping 6 prominent 2D image landmarks (Nose tip, Chin, Eye outer corners, Mouth corners) against a calibrated 3D generic anthropomorphic facial model, we solve the **Perspective-n-Point (PnP)** problem:

$$s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} \cdot \left( \mathbf{R} \cdot \mathbf{X}_{3D} + \mathbf{T} \right)$$

Using `cv2.solvePnP` and `cv2.Rodrigues`, we extract Euler angles:
* **Pitch**: Detects forward head nod (nodding off / micro-sleep).
* **Yaw**: Detects lateral head turn ($>25^\circ$) indicating driver distraction (looking at phones or passengers).

---

## 🚀 Quick Start

### 1. Installation
Ensure Python 3.10+ is installed, then install the dependencies:

```bash
cd driver-fatigue-monitor
pip install -r requirements.txt
```

### 2. Run High-Performance Desktop HUD (Recommended)
Launches the low-latency OpenCV window with real-time HUD telemetry:

```bash
python run_desktop.py
```

#### Controls:
* **`r`** : Recalibrate personalized baseline facial geometry.
* **`m`** : Mute / unmute audio warning alerts.
* **`s`** : Save instant HUD screenshot.
* **`q`** or **`ESC`** : Exit application.

### 3. Run Interactive Web Dashboard (Streamlit)
Launches a browser dashboard with interactive sensitivity sliders and telemetry charts:

```bash
streamlit run app_streamlit.py
```

---

## 📁 Repository Structure

```
driver-fatigue-monitor/
├── src/
│   ├── __init__.py
│   ├── landmarks.py        # 468 MediaPipe FaceMesh indices & 3D face model
│   ├── metrics.py          # Vectorized EAR, MAR, and solvePnP head pose math
│   ├── detector.py         # Self-calibrating temporal fatigue state machine
│   └── alert.py            # Non-blocking audio manager & modern HUD overlay
├── tests/
│   └── test_metrics.py     # Automated unit tests for mathematical accuracy
├── run_desktop.py          # Desktop runner entry point
├── app_streamlit.py        # Streamlit web application
├── requirements.txt        # Verified dependencies
└── README.md               # Technical documentation
```

---

## 👩‍💻 Author
**Megha Bhandari**  
B.Tech Information Technology, Maharaja Surajmal Institute of Technology  
[GitHub Profile](https://github.com/megha06102004)
