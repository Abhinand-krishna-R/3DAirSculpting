🎨 3DAirSculpt: Gesture-Controlled 3D Sculpting

3DAirSculpt is a computer vision–powered 3D sculpting environment that transforms 2D hand movements into solid 3D geometry in real-time. By leveraging MediaPipe for hand tracking and NumPy/SciPy for computational geometry, it allows users to sketch in thin air and generate interactive 3D objects—no mouse, no controller, just your hands.

🚀 Key Features

✍ Air Sketching: Draw paths using your index finger with real-time landmark tracking and instant visual feedback.

🧠 Path Smoothing: Integrated B-Spline smoothing via SciPy's UnivariateSpline to eliminate hand jitter and produce fluid, professional curves.

🔺 2D to 3D Solidification: Uses a custom Ear-Clipping Triangulation algorithm to convert arbitrary 2D polygons into closed, solid 3D manifold geometry.

🖐 Gesture-Based Controls: * ☝ Index Finger → Sketch path

🖐 Open Palm → Solidify shape into 3D

✌ Two Fingers → Orbit / Rotate scene

🤟 Three Fingers → Cycle color palettes

✊ Fist → Clear canvas

🎥 Custom 3D Engine: A built-in software rasterizer featuring flat shading, directional lighting, Z-depth sorting, and matrix-based 3D transformations.

🛠 Tech Stack

Library

Role

OpenCV

Video capture and high-performance 2D rendering

MediaPipe

High-fidelity hand landmark detection

NumPy

Linear algebra, 3D math, and vertex transformations

SciPy

Spline interpolation for path smoothing

🎮 Getting Started

Prerequisites

Ensure you have Python 3.x installed.

Installation

Clone the repository:

git clone [https://github.com/your-username/3DAirSculpt.git](https://github.com/your-username/3DAirSculpt.git)
cd 3DAirSculpt

Running the App

python 3dAirSculpt.py


⚙ Configuration

Adjust the sculpting experience by modifying parameters in the main script:

Parameter

Default

Description

GESTURE_HOLD_FRAMES

10

Frame duration required to trigger an action

HAND_POSITION_SMOOTHING

0.6

EMA factor for jitter control

BSPLINE_SMOOTHING

True

Toggle for curve refinement

🧩 Technical Highlights

Real-Time Pipeline: Low-latency hand tracking utilizing MediaPipe's lightweight models.

Jitter Mitigation: Uses Exponential Moving Average (EMA) filtering combined with B-spline interpolation for a "stable" feel.

Computational Geometry: Custom implementation of polygon triangulation to handle complex, concave shapes.

Graphics Pipeline: Fully manual 3D projection (perspective division) and rotation matrix application without external GPU-heavy engines.

📌 Project Focus

Computer Vision (HCI)

Computational Geometry

Real-Time Graphics

Touchless Interfaces

Created with ❤️ by [Your Name/GitHub]
