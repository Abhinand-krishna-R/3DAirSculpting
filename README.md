🎨 3DAirSculpt Pro

Gesture-Controlled Real-Time 3D Sculpting

AirSculpt Pro is a computer vision–powered 3D sculpting system that transforms 2D hand movements into solid 3D geometry in real time.

Using MediaPipe for hand tracking and NumPy/SciPy for geometric processing, users can sketch shapes in the air and convert them into interactive 3D objects — no mouse, no controller, just hand gestures.

🚀 Features
✍ Air Sketching

Draw 2D paths using your index finger. The system captures hand landmarks and tracks motion in real time.

🧠 B-Spline Smoothing

Applies UnivariateSpline interpolation (SciPy) to convert unstable hand paths into smooth, elegant curves.

🔺 2D → 3D Solidification

Implements a custom ear-clipping triangulation algorithm to convert arbitrary polygons into closed 3D meshes.

🖐 Gesture-Driven Controls
Gesture	Action
☝ Index Finger	Sketch 2D path
🖐 Open Palm	Solidify into 3D object
✌ Two Fingers	Orbit / Rotate scene
🤟 Three Fingers	Cycle color palette
✊ Fist	Clear scene
🎥 Real-Time Rendering Engine

Custom software rasterizer

Flat shading

Basic directional lighting

Z-depth sorting

Manual matrix-based 3D projection

No external 3D engine is used — all transformations are computed with NumPy.

🛠 Tech Stack

Python 3.x

OpenCV – Webcam capture & 2D interface rendering

MediaPipe – Machine learning–based hand landmark detection

NumPy – Linear algebra for 3D transformations

SciPy – Spline interpolation for smoothing

🎮 How to Run
python 3dAirSculpt.py

Make sure required dependencies are installed:

pip install -r requirements.txt
⚙ Configuration

Adjust these parameters inside the script:

GESTURE_HOLD_FRAMES – Duration required to trigger actions

HAND_POSITION_SMOOTHING – Controls EMA jitter filtering

BSPLINE_SMOOTHING – Controls curve smoothness

🧩 Technical Highlights

Real-time hand landmark processing

Exponential Moving Average (EMA) filtering

B-spline curve interpolation

Custom polygon triangulation

3D mesh generation from 2D sketches

Perspective projection & rotation matrices
