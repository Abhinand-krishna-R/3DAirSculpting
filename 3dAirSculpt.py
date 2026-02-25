import cv2
import mediapipe as mp
import numpy as np
import time
import math
import random
from collections import deque
from scipy.interpolate import UnivariateSpline

# ==========================
# CONFIGURATION
# ==========================

EMA_ALPHA = 0.22           
GESTURE_HOLD_FRAMES = 35  
SHAPE_CLOSE_THRESHOLD = 50 

# Gesture Smoothing Parameters
GESTURE_CONFIDENCE_THRESHOLD = 0.65
GESTURE_BUFFER_SIZE = 5
HAND_POSITION_SMOOTHING = 0.28
ORBIT_SENSITIVITY = 200

# Fist/Palm Detection
FINGER_CURL_THRESHOLD = 0.05     # How much a finger must bend from extended position
MIN_CURL_FOR_FIST = 4            # All 5 fingers must be mostly curled (thumb + 4 fingers)

# Point Resampling (B-Spline smoothing)
BSPLINE_POINTS = 150             # Number of resampled points
BSPLINE_SMOOTHING = 0.1          # Smoothing factor (lower = smoother)

# High-Contrast Solid Palette
COLORS = [
    { 'name': 'Teal', 'bgr': (180, 220, 100) },
    { 'name': 'Sculpt Blue', 'bgr': (246, 130, 59) },
    { 'name': 'Action Orange', 'bgr': (22, 115, 249) },
    { 'name': 'Emerald', 'bgr': (129, 185, 16) },
    { 'name': 'White', 'bgr': (255, 255, 255) }
]

LIGHT_DIR = np.array([-0.6, -1.0, -0.5])
LIGHT_DIR /= np.linalg.norm(LIGHT_DIR)

# ==========================
# ADVANCED POINT RESAMPLING
# ==========================

class PointResampler:
    """Uses B-Spline interpolation to smooth sketchy hand paths."""
    
    @staticmethod
    def resample_with_bspline(points, num_points=BSPLINE_POINTS, smoothing=BSPLINE_SMOOTHING):
        if len(points) < 4:
            return np.array(points, dtype=np.float32)
        
        pts = np.array(points, dtype=np.float32)
        diffs = np.diff(pts, axis=0)
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        cumsum = np.concatenate(([0], np.cumsum(distances)))
        
        if cumsum[-1] > 0:
            t = cumsum / cumsum[-1]
        else:
            return pts
        
        try:
            spl_x = UnivariateSpline(t, pts[:, 0], s=len(pts) * smoothing, k=min(3, len(pts)-1))
            spl_y = UnivariateSpline(t, pts[:, 1], s=len(pts) * smoothing, k=min(3, len(pts)-1))
            t_new = np.linspace(0, 1, num_points)
            resampled_x = spl_x(t_new)
            resampled_y = spl_y(t_new)
            return np.column_stack((resampled_x, resampled_y)).astype(np.float32)
        except:
            return PointResampler.resample_linear(pts, num_points)
    
    @staticmethod
    def resample_linear(points, num_points=BSPLINE_POINTS):
        pts = np.array(points, dtype=np.float32)
        diffs = np.diff(pts, axis=0)
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        cumsum = np.concatenate(([0], np.cumsum(distances)))
        if cumsum[-1] == 0: return pts
        t_new = np.linspace(0, cumsum[-1], num_points)
        resampled_x = np.interp(t_new, cumsum, pts[:, 0])
        resampled_y = np.interp(t_new, cumsum, pts[:, 1])
        return np.column_stack((resampled_x, resampled_y)).astype(np.float32)

# ==========================
# GESTURE CONFIDENCE SYSTEM
# ==========================

class GestureConfidence:
    def __init__(self, buffer_size=5):
        self.buffer = deque(maxlen=buffer_size)
        self.current_gesture = "IDLE"
        self.confidence_scores = {
            "IDLE": 0.0, "FIST": 0.0, "PALM": 0.0, "PALETTE": 0.0,
            "ORBIT": 0.0, "SKETCH": 0.0
        }
    
    def add_detection(self, gesture_name, confidence):
        self.buffer.append((gesture_name, confidence))
        self._update_scores()
    
    def _update_scores(self):
        self.confidence_scores = {g: 0.0 for g in self.confidence_scores}
        for gesture, confidence in self.buffer:
            if gesture in self.confidence_scores:
                self.confidence_scores[gesture] += confidence
        for key in self.confidence_scores:
            self.confidence_scores[key] /= len(self.buffer) if self.buffer else 1
    
    def get_dominant_gesture(self, threshold=GESTURE_CONFIDENCE_THRESHOLD):
        max_gesture = max(self.confidence_scores, key=self.confidence_scores.get)
        max_confidence = self.confidence_scores[max_gesture]
        if max_confidence >= threshold:
            return max_gesture, max_confidence
        return "IDLE", 0.0

# ==========================
# SMOOTH HAND TRACKING
# ==========================

class HandTracker:
    def __init__(self, alpha=HAND_POSITION_SMOOTHING):
        self.alpha = alpha
        self.smoothed_position = None
    
    def update(self, lm_array, frame_w, frame_h):
        raw_x, raw_y = lm_array[8].x, lm_array[8].y
        raw_pos = np.array([raw_x, raw_y])
        if self.smoothed_position is None:
            self.smoothed_position = raw_pos
        else:
            self.smoothed_position = self.alpha * raw_pos + (1 - self.alpha) * self.smoothed_position
        return (int(self.smoothed_position[0] * frame_w), int(self.smoothed_position[1] * frame_h))

# ==========================
# ENHANCED GESTURE DETECTION
# ==========================

class GestureDetector:
    def get_finger_curl(self, lm, finger_tip_idx, finger_pip_idx, finger_mcp_idx, wrist_idx=0):
        wrist = np.array([lm[wrist_idx].x, lm[wrist_idx].y, lm[wrist_idx].z])
        tip = np.array([lm[finger_tip_idx].x, lm[finger_tip_idx].y, lm[finger_tip_idx].z])
        mcp = np.array([lm[finger_mcp_idx].x, lm[finger_mcp_idx].y, lm[finger_mcp_idx].z])
        tip_to_wrist = np.linalg.norm(tip - wrist)
        mcp_to_wrist = np.linalg.norm(mcp - wrist)
        if mcp_to_wrist > 0:
            curl = 1.0 - (tip_to_wrist / mcp_to_wrist)
            return np.clip(curl, 0.0, 1.0)
        return 0.0
    
    def analyze_hand(self, lm, hand_size):
        thumb_curl = self.get_finger_curl(lm, 4, 3, 2, 0)
        index_curl = self.get_finger_curl(lm, 8, 7, 6, 0)
        middle_curl = self.get_finger_curl(lm, 12, 11, 10, 0)
        ring_curl = self.get_finger_curl(lm, 16, 15, 14, 0)
        pinky_curl = self.get_finger_curl(lm, 20, 19, 18, 0)
        
        idx_extended = lm[8].y < lm[6].y - 0.02
        mid_extended = lm[12].y < lm[10].y - 0.02
        rng_extended = lm[16].y < lm[14].y - 0.02
        pky_extended = lm[20].y < lm[18].y - 0.02
        thumb_extended = lm[4].x < lm[3].x if lm[0].x > 0.5 else lm[4].x > lm[3].x
        
        curled_fingers = sum([
            index_curl > FINGER_CURL_THRESHOLD,
            middle_curl > FINGER_CURL_THRESHOLD,
            ring_curl > FINGER_CURL_THRESHOLD,
            pinky_curl > FINGER_CURL_THRESHOLD,
            thumb_curl > FINGER_CURL_THRESHOLD
        ])
        
        is_fist = curled_fingers >= MIN_CURL_FOR_FIST
        is_palm = idx_extended and mid_extended and rng_extended and pky_extended and thumb_extended
        
        gestures = {}
        if is_fist:
            avg_curl = (index_curl + middle_curl + ring_curl + pinky_curl) / 4.0
            gestures["FIST"] = 0.90 + (avg_curl * 0.08)
        
        if is_palm:
            gestures["PALM"] = 0.92
        
        if idx_extended and mid_extended and rng_extended and not pky_extended and thumb_extended:
            gestures["PALETTE"] = 0.88
        
        if idx_extended and mid_extended and not rng_extended and not pky_extended:
            gestures["ORBIT"] = 0.85
        
        if idx_extended and not mid_extended and not rng_extended and not pky_extended and index_curl < 0.3:
            gestures["SKETCH"] = 0.92
        
        if not gestures:
            gestures["IDLE"] = 1.0
        
        return gestures

# ==========================
# GEOMETRY ENGINE
# ==========================

def triangulate_polygon(pts):
    indices = list(range(len(pts)))
    triangles = []
    area = 0
    for i in range(len(pts)):
        p1, p2 = pts[i], pts[(i + 1) % len(pts)]
        area += (p2[0] - p1[0]) * (p2[1] + p1[1])
    if area > 0: indices = indices[::-1]
    limit = 0
    while len(indices) > 3 and limit < 600:
        limit += 1
        for i in range(len(indices)):
            prev, curr, nxt = indices[i-1], indices[i], indices[(i+1)%len(indices)]
            A, B, C = pts[prev], pts[curr], pts[nxt]
            vec1, vec2 = B - A, C - B
            if (vec1[0]*vec2[1] - vec1[1]*vec2[0]) < 0: continue 
            is_ear = True
            for j in range(len(indices)):
                idx = indices[j]
                if idx in (prev, curr, nxt): continue
                P = pts[idx]
                v0, v1, v2 = C - A, B - A, P - A
                d00, d01, d02 = np.dot(v0,v0), np.dot(v0,v1), np.dot(v0,v2)
                d11, d12 = np.dot(v1,v1), np.dot(v1,v2)
                inv_denom = 1 / (d00 * d11 - d01 * d01 + 1e-9)
                u, v = (d11*d02 - d01*d12)*inv_denom, (d00*d12 - d01*d02)*inv_denom
                if (u >= 0) and (v >= 0) and (u + v < 1):
                    is_ear = False; break
            if is_ear:
                triangles.append((prev, curr, nxt))
                indices.pop(i); break
    if len(indices) == 3: triangles.append(tuple(indices))
    return triangles

class WorldObject:
    def __init__(self, vertices, faces, color):
        self.vertices = np.array(vertices, dtype=np.float32)
        self.faces = faces 
        self.color = color
        self.position = np.array([0, 0, 800], dtype=np.float32)
        self.rotation = np.array([0, 0, 0], dtype=np.float32)
        self.scale = 1.0
        self.is_selected = False

    def get_transformed_data(self, global_rot):
        rx, ry, rz = np.radians(self.rotation + global_rot)
        Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
        Ry = np.array([[np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]])
        Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])
        return (self.vertices * self.scale) @ (Rz @ Ry @ Rx).T + self.position

def project_points(points, w, h, fov=1200):
    cx, cy = w // 2, h // 2
    factor = fov / (points[:, 2:3] + 1e-5)
    px = (points[:, 0:1] * factor + cx).astype(np.int32)
    py = (points[:, 1:2] * factor + cy).astype(np.int32)
    return np.hstack((px, py))

# ==========================
# AIRSCULPT PRO STREAMLINED
# ==========================

class AirSculptPro:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False, 
            max_num_hands=1, 
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        self.scene, self.stroke_points = [], []
        self.temp_layer, self.prev_smooth = None, None
        self.tracking_state = "LOST"
        self.finalize_buffer, self.clear_buffer = 0, 0
        self.color_index, self.color_cooldown = 0, 0
        self.global_rotation = np.array([0.0, 0.0, 0.0])
        self.mode, self.hold_percent = "IDLE", 0
        self.active_mode = "IDLE"
        
        self.gesture_confidence = GestureConfidence(buffer_size=GESTURE_BUFFER_SIZE)
        self.hand_tracker = HandTracker(alpha=HAND_POSITION_SMOOTHING)
        self.gesture_detector = GestureDetector()
        self.point_resampler = PointResampler()
        
        self.last_orbit_pos = None
        self.orbit_smoothing = 0.15
        self.target_rotation = np.array([0.0, 0.0, 0.0])

    def process_frame(self, frame):
        h, w, _ = frame.shape
        if self.temp_layer is None: self.temp_layer = np.zeros_like(frame)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        self.hold_percent = 0

        if not results.multi_hand_landmarks:
            self.tracking_state = "LOST"
            self.mode = "WAITING"
            self.prev_smooth = None
            self.active_mode = "IDLE"
            self.last_orbit_pos = None
            return self.render_scene(frame)

        self.tracking_state = "LOCKED"
        lm = results.multi_hand_landmarks[0].landmark
        h_size = np.linalg.norm(np.array([lm[0].x - lm[9].x, lm[0].y - lm[9].y]))
        self.prev_smooth = self.hand_tracker.update(lm, w, h)
        gesture_scores = self.gesture_detector.analyze_hand(lm, h_size)
        
        for gesture, confidence in gesture_scores.items():
            self.gesture_confidence.add_detection(gesture, confidence)
        
        dominant_gesture, confidence = self.gesture_confidence.get_dominant_gesture()
        
        if dominant_gesture != self.active_mode:
            self.active_mode = dominant_gesture
            self.finalize_buffer = 0
            self.clear_buffer = 0
            self.last_orbit_pos = None

        if self.active_mode != "PALM": self.finalize_buffer = 0
        if self.active_mode != "FIST": self.clear_buffer = 0

        # ==================== GESTURE HANDLERS ====================
        
        if self.active_mode == "ORBIT":
            self.mode = "ORBITING"
            raw_x, raw_y = lm[8].x, lm[8].y
            if self.last_orbit_pos is not None:
                dx = (raw_x - self.last_orbit_pos[0]) * ORBIT_SENSITIVITY
                dy = (raw_y - self.last_orbit_pos[1]) * ORBIT_SENSITIVITY
                self.target_rotation[1] += dx
                self.target_rotation[0] -= dy
                self.global_rotation = self.orbit_smoothing * self.target_rotation + (1 - self.orbit_smoothing) * self.global_rotation
            self.last_orbit_pos = (raw_x, raw_y)

        elif self.active_mode == "FIST":
            self.mode = "PURGING"
            self.clear_buffer += 1
            self.hold_percent = min(100, int((self.clear_buffer / GESTURE_HOLD_FRAMES) * 100))
            if self.clear_buffer >= GESTURE_HOLD_FRAMES:
                self.scene, self.temp_layer, self.stroke_points = [], np.zeros_like(frame), []

        elif self.active_mode == "PALM":
            if len(self.stroke_points) > 10:
                self.mode = "SOLIDIFYING"
                self.finalize_buffer += 1
                self.hold_percent = min(100, int((self.finalize_buffer / GESTURE_HOLD_FRAMES) * 100))
                if self.finalize_buffer >= GESTURE_HOLD_FRAMES:
                    self.bake_3d(w, h)
                    self.stroke_points, self.temp_layer = [], np.zeros_like(frame)
            else:
                self.mode = "READY"

        elif self.active_mode == "PALETTE":
            self.mode = "COLOR"
            if time.time() - self.color_cooldown > 0.8:
                self.color_index = (self.color_index + 1) % len(COLORS)
                self.color_cooldown = time.time()

        elif self.active_mode == "SKETCH":
            self.mode = "SKETCHING"
            if self.stroke_points:
                cv2.line(self.temp_layer, self.stroke_points[-1], self.prev_smooth, COLORS[self.color_index]['bgr'], 6, cv2.LINE_AA)
            self.stroke_points.append(self.prev_smooth)

        else:
            self.mode = "IDLE"

        return self.render_scene(frame)

    def bake_3d(self, w, h):
        if len(self.stroke_points) < 5: return
        pts_resampled = self.point_resampler.resample_with_bspline(self.stroke_points)
        pts_int = pts_resampled.astype(np.int32)
        peri = cv2.arcLength(pts_int, True)
        if peri == 0: return
        approx = cv2.approxPolyDP(pts_int, 0.005 * peri, True).reshape(-1, 2)
        if len(approx) < 3: return
        
        approx_float = approx.astype(np.float32)
        dists = np.sqrt(np.sum(np.diff(approx_float, axis=0)**2, axis=1))
        cum = np.concatenate(([0], np.cumsum(dists)))
        if cum[-1] == 0: return
        
        resampled = np.array([
            np.interp(np.linspace(0, cum[-1], 150), cum, approx_float[:, 0]),
            np.interp(np.linspace(0, cum[-1], 150), cum, approx_float[:, 1])
        ]).T
        
        centroid = np.mean(resampled, axis=0)
        depth = np.clip(len(self.stroke_points) * 0.7, 100, 300)
        n = len(resampled)
        rel_pts = resampled - centroid
        
        v = []
        for p in rel_pts: v.append([p[0], p[1], -depth/2])
        for p in rel_pts: v.append([p[0], p[1], depth/2])
        
        triangles = triangulate_polygon(rel_pts)
        f = []
        for t in triangles: f.append((t[0], t[1], t[2]))
        for t in triangles: f.append((t[0]+n, t[2]+n, t[1]+n))
        for i in range(n):
            nxt = (i+1) % n
            f.extend([(i, nxt, nxt+n), (i, nxt+n, i+n)])
        
        obj = WorldObject(v, f, COLORS[self.color_index]['bgr'])
        obj.position = np.array([centroid[0]-w/2, centroid[1]-h/2, 800])
        self.scene.append(obj)

    def render_scene(self, frame):
        all_faces, h, w = [], frame.shape[0], frame.shape[1]
        for obj in self.scene:
            v_world = obj.get_transformed_data(self.global_rotation)
            pts_2d = project_points(v_world, w, h)
            for face in obj.faces:
                v0, v1, v2 = v_world[face[0]], v_world[face[1]], v_world[face[2]]
                normal = np.cross(v1 - v0, v2 - v0)
                n_len = np.linalg.norm(normal)
                if n_len < 1e-5: continue
                normal /= n_len
                if np.dot(normal, -v0) < 0: continue 
                diffuse = np.clip(np.dot(normal, -LIGHT_DIR), 0.0, 1.0)
                intensity = np.clip(0.3 + diffuse * 0.7, 0.1, 1.0)
                all_faces.append({
                    'pts': np.array([pts_2d[i] for i in face], np.int32), 
                    'z': np.mean(v_world[np.array(face), 2]), 
                    'color': tuple([int(c*intensity) for c in obj.color])
                })
        
        all_faces.sort(key=lambda x: x['z'], reverse=True)
        for f in all_faces:
            cv2.fillPoly(frame, [f['pts']], f['color'])

        mask = cv2.threshold(cv2.cvtColor(self.temp_layer, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)[1]
        frame[mask > 0] = self.temp_layer[mask > 0]
        self.draw_ui(frame, w, h)
        return frame

    def draw_ui(self, frame, w, h):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 55), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, f"AIRSCULPT PRO // {self.mode}", (w//2-160, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2, cv2.LINE_AA)
        
        for i, c in enumerate(COLORS):
            py = 115+(i*48)
            if i == self.color_index: cv2.circle(frame, (48, py), 20, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(frame, (48, py), 15, c['bgr'], -1, cv2.LINE_AA)
            
        if self.hold_percent > 0:
            bw, bx = 300, (w-300)//2
            cv2.rectangle(frame, (bx, h-85), (bx+bw, h-80), (45, 45, 50), -1)
            fill = int((self.hold_percent/100)*bw)
            cv2.rectangle(frame, (bx, h-85), (bx+fill, h-80), (255, 255, 255), -1)
            
        gx, gy = w - 240, 80
        cv2.rectangle(frame, (gx, gy), (w - 20, gy + 160), (15, 15, 20), -1)
        cv2.rectangle(frame, (gx, gy), (w - 20, gy + 160), (60, 60, 60), 1)
        guide = [
            ("SKETCH", "INDEX UP"), 
            ("ORBIT", "2-FINGERS"), 
            ("COLOR", "3-FINGERS"), 
            ("SOLID", "OPEN PALM"), 
            ("PURGE", "FIST (HOLD)")
        ]
        for i, (f, g) in enumerate(guide):
            cv2.putText(frame, f, (gx+15, gy+30+i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140,140,140), 1, cv2.LINE_AA)
            cv2.putText(frame, g, (gx+95, gy+30+i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1, cv2.LINE_AA)

if __name__ == "__main__":
    sculptor = AirSculptPro()
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        cv2.imshow("AirSculpt Pro", sculptor.process_frame(cv2.flip(frame, 1)))
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()