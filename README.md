AirSculpt Pro

AirSculpt Pro is an advanced, gesture-controlled 3D sculpting environment that translates 2D hand movements into solid, manifold 3D geometry in real-time. By leveraging MediaPipe for high-fidelity hand tracking and NumPy/SciPy for geometric processing, it allows users to "sketch" in the air and solidify those sketches into interactive 3D objects.

🚀 Key Features

B-Spline Smoothing: Uses UnivariateSpline interpolation to convert shaky hand paths into elegant, fluid curves.

Ear-Clipping Triangulation: A custom geometry engine that converts arbitrary 2D polygons into solid 3D manifolds with closed caps.

Gesture-Driven UI:

Sketch (Index Finger): Draw 2D paths on the temporary canvas.

Solidify (Open Palm): Hold an open palm to "bake" your 2D sketch into a 3D object.

Orbit (Two Fingers): Rotate the entire 3D scene to view your creation from different angles.

Color Cycle (Three Fingers): Swap between high-contrast professional palettes.

Purge (Fist): Hold a fist to clear the scene and start over.

Real-time Rendering: A custom software rasterizer featuring flat shading, simple directional lighting, and Z-depth sorting.

🛠 Tech Stack

Python 3.x

OpenCV: Video capture and 2D UI rendering.

MediaPipe: ML-based hand landmark detection.

NumPy: Linear algebra for 3D transformations and projections.

SciPy: Advanced spline interpolation for path smoothing.

🎮 How to Use

Run the script:

python 3dAirSculpt.py


Sketching: Raise your index finger and move it across the screen. You will see a 2D line following your finger.

Solidifying: Once you are happy with the shape, open your palm. A progress bar will appear at the bottom. Once full, the shape becomes a 3D object.

Orbiting: Use two fingers (Index and Middle) to "grab" the air and rotate the world.

Clearing: To start a new project, close your hand into a fist and hold until the progress bar completes.

⚙️ Configuration

You can tune the experience in the CONFIGURATION section of the script:

GESTURE_HOLD_FRAMES: Adjust how long you need to hold a gesture to trigger "Solidify" or "Purge".

HAND_POSITION_SMOOTHING: Control the EMA (Exponential Moving Average) filter for hand jitter.

BSPLINE_SMOOTHING: Adjust how much the spline "ignores" small hand tremors.

⚖️ License

MIT License - Feel free to use and modify for your own creative coding projects!
