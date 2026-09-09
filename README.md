# 🚗 Real-Time Driver Fatigue & Distraction Monitoring System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow%2FKeras-CNN%20Classification-FF6F00.svg)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-FaceMesh%203D-orange.svg)](https://developers.google.com/mediapipe)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An enterprise-grade, multi-modal driver safety platform leveraging a **Convolutional Neural Network (CNN)** for real-time eye-state classification on live video frames, combined with 3D Head Pose tracking (`cv2.solvePnP`) and Eye Aspect Ratio (EAR) geometry.

---

## 🔬 Deep Learning Architecture (Eye-State CNN)

To deliver robust eye-state classification (`Open` vs. `Closed`) across varying lighting, glasses, and head orientations, the system extracts eye bounding regions from video frames and passes them to a custom 3-block 2D Convolutional Neural Network:

```
Input Video Frame (1080p / 720p @ 60 FPS)
                   │
                   ▼
     [Facial Landmark Tracking]
     - Extracts Left & Right Eye Bounding Boxes
     - Dynamic Margin Padding & Normalization
                   │
                   ▼
       [Eye Image Crop: (64, 64, 3)]
                   │
                   ▼
    ┌──────────────────────────────────────────┐
    │          Convolutional Layers            │
    │  • Conv2D(32, 3x3) + BatchNorm + MaxPool │
    │  • Conv2D(64, 3x3) + BatchNorm + MaxPool │
    │  • Conv2D(128, 3x3) + BatchNorm + MaxPool│
    └──────────────────────────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────────────┐
    │          Dense Classifier Head           │
    │  • Flatten + Dense(128, ReLU) + Dropout  │
    │  • Dense(1, Sigmoid)                     │
    └──────────────────────────────────────────┘
                   │
                   ▼
      P(Open) vs P(Closed) Probability
```

### CNN Validation Benchmarks:
* **Validation Accuracy:** **92.78%**
* **ROC-AUC Score:** **0.9141**
* **Precision (Open Eyes):** **0.99**
* **Recall (Closed Eyes):** **0.99**
* **Inference Latency:** **< 2.5 ms** per crop on CPU

---

## ⚡ Multi-Modal Sensor Fusion Pipeline

The monitoring pipeline fuses three complementary detection modalities:
1. **CNN Eye-State Classification:** Deep visual feature extraction for eyelid closure detection.
2. **Eye Aspect Ratio (EAR):** Geometric Euclidean distance ratio validating micro-sleep events.
3. **3D Head Pose (`cv2.solvePnP`):** Computes Pitch, Yaw, and Roll using Perspective-n-Point geometry on 3D canonical facial model vertices to catch head nodding and distraction.

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/megha06102004/driver-fatigue-monitor.git
cd driver-fatigue-monitor
pip install -r requirements.txt
```

### 2. Run Standalone Desktop Application
```bash
python run_desktop.py
```

### 3. Retrain the CNN Model (Optional)
To retrain the Convolutional Neural Network from raw eye image crops:
```bash
python train_cnn.py
```

### 4. Run Automated Unit Tests
```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## 📁 Repository Structure

```
driver-fatigue-monitor/
├── src/
│   ├── __init__.py
│   ├── cnn_classifier.py    # CNN architecture, eye crop extraction & inference
│   ├── landmarks.py         # 468 MediaPipe FaceMesh indices & 3D face model
│   ├── metrics.py           # Vectorized EAR, MAR, and cv2.solvePnP 3D pose
│   ├── alert.py             # Non-blocking audio alert thread & HUD visualizer
│   └── detector.py          # Multi-modal fusion state machine (CNN + EAR + Pose)
├── models/
│   ├── eye_state_cnn.keras  # Trained TensorFlow / Keras CNN model weights
│   ├── cnn_metadata.json    # Calibrated decision threshold & metrics
│   └── cnn_training_curves.png # Loss & accuracy training curve plots
├── data/
│   └── eye_dataset/         # Labeled eye image crops (Closed & Open)
├── tests/
│   ├── test_metrics.py      # Geometry and audio threading tests
│   └── test_cnn.py          # CNN architecture and inference unit tests
├── train_cnn.py             # Complete CNN training & validation pipeline
├── run_desktop.py           # 60+ FPS desktop webcam runner
├── app_streamlit.py         # Web dashboard runner
├── requirements.txt         # Verified dependencies
└── README.md                # Technical documentation
```

---

## 👩‍💻 Author
**Megha Bhandari**  
B.Tech Information Technology, Maharaja Surajmal Institute of Technology  
[GitHub Profile](https://github.com/megha06102004)
