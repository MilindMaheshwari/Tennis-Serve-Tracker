# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: tennis
#     language: python
#     name: python3
# ---

# %%
# %pip install ultralytics

# %%

# %pip install git+https://github.com/facebookresearch/segment-anything-2.git

# %% [markdown]
# # Just using YOLO (opening webcam)

# %%
import cv2
import time


def display_YOLO_only(yolo_model):
    # Run inference on webcam
    # Note: This will open a separate window to display the video feed
    cap = cv2.VideoCapture(0)

    prev_frame_time = time.perf_counter()

    if not cap.isOpened():
        print("Error: Could not open webcam")
    else:
        print("Webcam opened successfully!")
        print("A separate window will open showing the detection results.")
        print("Press 'q' in the video window to quit, or interrupt the kernel to stop")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                new_frame_time = time.perf_counter()
                fps = 1 / (new_frame_time - prev_frame_time)
                prev_frame_time = new_frame_time
                
                # Run inference
                results = yolo_model(frame)

                cv2.putText(frame, f"FPS: {fps}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Draw results on frame
                annotated_frame = results[0].plot()
                
                # Display the frame in a separate window
                cv2.imshow('Tennis Ball Detection', annotated_frame)
                cv2.plot
                
                # Break loop on 'q' key press (make sure the video window is focused)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Stopping...")
                    break
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            # Always clean up, even if interrupted
            cap.release()
            cv2.destroyAllWindows()
            print("Webcam releas=ed and windows closed")



# %% [markdown]
# # YOLO resolution changing

# %%
import cv2
import time
import math
from ultralytics import YOLO

# Load model (assuming you have this defined elsewhere, or load it here)
# yolo_model = YOLO("yolov8n.pt") 

# --- CONFIGURATION ---
# Downsize inference to this width. 
# 640 is standard. 320 is faster. 1920 is native (slow).
YOLO_WIDTH = 3840 

cap = cv2.VideoCapture(0)

# Use perf_counter for high precision timing
prev_frame_time = time.perf_counter()

if not cap.isOpened():
    print("Error: Could not open webcam")
else:
    print("Webcam opened successfully!")
    print(f"Inference running at {YOLO_WIDTH}px width.")
    print("Press 'q' in the video window to quit.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

                    
            # Flip frame horizontally (mirror effect)
            frame = cv2.flip(frame, 1)


            # --- 1. RESIZE FOR INFERENCE ---
            # Get original dimensions
            orig_h, orig_w = frame.shape[:2]
            
            # Calculate new height to maintain aspect ratio
            yolo_h = int(orig_h * (YOLO_WIDTH / orig_w))
            
            # Create the small image (CPU operation, very fast)
            frame_small = cv2.resize(frame, (YOLO_WIDTH, yolo_h))

            # --- 2. RUN INFERENCE ---
            # Run YOLO on the SMALL frame
            results = yolo_model(frame_small, verbose=False)

            # --- 3. SCALE & DRAW ON ORIGINAL FRAME ---
            # Calculate how much we need to multiply the coords by
            x_scale = orig_w / YOLO_WIDTH
            y_scale = orig_h / yolo_h

            # Iterate through detections
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Filter: Only draw if confidence > 0.5 (optional)
                    if box.conf[0] > 0.5:
                        # Get coords from SMALL image
                        x1_s, y1_s, x2_s, y2_s = box.xyxy[0].cpu().numpy()
                        
                        # Scale up to ORIGINAL image
                        x1 = int(x1_s * x_scale)
                        y1 = int(y1_s * y_scale)
                        x2 = int(x2_s * x_scale)
                        y2 = int(y2_s * y_scale)
                        
                        # Get Class Name
                        cls_id = int(box.cls[0])
                        cls_name = yolo_model.names[cls_id]
                        conf = float(box.conf[0])

                        # Draw Box on HD Frame
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw Label
                        label = f"{cls_name} {conf:.2f}"
                        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(frame, (x1, y1 - t_size[1] - 5), (x1 + t_size[0], y1), (0, 255, 0), -1)
                        cv2.putText(frame, label, (x1, y1 - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # --- 4. CALCULATE FPS ---
            # Calculate after inference to include processing time
            new_frame_time = time.perf_counter()
            time_diff = new_frame_time - prev_frame_time
            fps = 1 / time_diff if time_diff > 0 else 0
            prev_frame_time = new_frame_time

            # Draw FPS
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # --- 5. DISPLAY ---
            # Show the original HD frame
            cv2.imshow('Tennis Ball Detection (HD Feed)', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Stopping...")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam released and windows closed")

# %% [markdown]
# # Import sam2 (not ousing anymore)

# %%
import torch
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 1. Define paths
current_dir = os.getcwd()
local_config_path = os.path.join(current_dir, "sam2", "sam2_hiera_t.yaml")
checkpoint_path = os.path.join(current_dir, "sam2", "sam2_hiera_tiny.pt")

# 2. Verify files exist
if not os.path.exists(local_config_path):
    raise FileNotFoundError(f"Config not found at: {local_config_path}")
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

print("Loading SAM2 model...")

# 3. Build video predictor in one step (builds model + creates predictor)
# This is simpler than: build_sam2() then SAM2VideoPredictor()
sam2_model = build_sam2(
    config_file=local_config_path,
    ckpt_path=checkpoint_path,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

sam2_predictor = SAM2ImagePredictor(sam2_model)

print("SAM2 model loaded successfully!")




# %% [markdown]
# # INITIALIZE YOLO

# %%
from ultralytics import YOLO
import cv2
import time
import numpy as np

yolo_model = YOLO("weights/best (3).pt")
TRACKER_TYPE = "MOSSE"          # CSRT is best for accuracy on CPU

# %%
# --- Trajectory Straightness Helpers ---
# Usage:
# 1) Run this cell to define helpers.
# 2) Create `monitor = TrajectoryMonitor(min_len=5)` once (e.g., before your tracking loop).
# 3) Inside your tracking loop, when the tracker successfully returns a box, call:
#       cx = int(x + w/2); cy = int(y + h/2)
#       monitor.add_point(cx, cy)
#    and after drawing the bbox call `monitor.draw(frame)` to overlay the fitted line & score.
# 4) When tracking ends (tracker reset/failed), call:
#       score, metrics = monitor.finalize()
#    to compute and retrieve the final straightness score for the segment.

import numpy as np
import cv2
from typing import Tuple, Dict, Optional


def fit_line_pca(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a 2D line using PCA. Returns (point_on_line, direction_vector)."""
    pts = np.asarray(points, dtype=float)
    assert pts.ndim == 2 and pts.shape[1] == 2
    mean = pts.mean(axis=0)
    centered = pts - mean
    # SVD: principal direction is first right-singular vector
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    return mean, direction


def point_line_distances(points: np.ndarray, p0: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Return perpendicular distances from points to the line (p0 + t * direction)."""
    pts = np.asarray(points, dtype=float)
    d = direction / (np.linalg.norm(direction) + 1e-12)
    diffs = pts - p0
    proj_len = np.dot(diffs, d)
    proj = np.outer(proj_len, d)
    perp = diffs - proj
    dists = np.linalg.norm(perp, axis=1)
    return dists


def compute_straightness_score(points: np.ndarray, k: float = 5.0) -> Tuple[float, Dict]:
    """Compute straightness score (0-100) and return metrics.

    - Uses PCA fit for the line.
    - mean_dev normalized by trajectory length to be scale invariant.
    - R² (variance explained) combined with normalized mean to form score.
    k controls sensitivity to mean deviation.
    """
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 2:
        return 0.0, {"mean_dev": 0.0, "max_dev": 0.0, "norm_mean": 0.0, "r2": 0.0}

    p0, direction = fit_line_pca(pts)
    dists = point_line_distances(pts, p0, direction)
    mean_dev = float(np.mean(dists))
    max_dev = float(np.max(dists))

    # Project onto line to compute length
    d = direction / (np.linalg.norm(direction) + 1e-12)
    proj_len = np.dot(pts - p0, d)
    length = float(proj_len.max() - proj_len.min())

    norm_mean = mean_dev / (length + 1e-6)

    # R²-like measure: variance explained by projection onto the line
    total_var = np.sum((pts - pts.mean(axis=0)) ** 2)
    reconstructed = p0 + np.outer(proj_len, d)
    residual_var = np.sum((pts - reconstructed) ** 2)
    r2 = float(max(0.0, 1.0 - residual_var / (total_var + 1e-12)))

    norm_mean_clipped = float(min(1.0, norm_mean * k))
    score = 100.0 * (0.6 * r2 + 0.4 * (1.0 - norm_mean_clipped))
    score = float(np.clip(score, 0.0, 100.0))

    metrics = {
        "mean_dev": mean_dev,
        "max_dev": max_dev,
        "norm_mean": norm_mean,
        "r2": r2,
    }
    return score, metrics


class TrajectoryMonitor:
    """Collects center points, computes straightness, and draws overlay.

    Methods:
      - add_point(x,y)
      - finalize() -> (score, metrics)  # call when segment ends
      - draw(frame)  # overlays points, fitted line and score on frame
    """

    def __init__(self, min_len: int = 5, max_points: int = 200):
        self.points = []
        self.min_len = min_len
        self.max_points = max_points
        self.last_score: Optional[float] = None
        self.last_metrics: Optional[Dict] = None

    def add_point(self, x: float, y: float) -> None:
        self.points.append((float(x), float(y)))
        if len(self.points) > self.max_points:
            # keep a rolling buffer
            self.points.pop(0)

    def reset(self) -> None:
        self.points = []

    def finalize(self) -> Tuple[Optional[float], Optional[Dict]]:
        if len(self.points) >= self.min_len:
            pts = np.array(self.points)
            score, metrics = compute_straightness_score(pts)
            self.last_score = score
            self.last_metrics = metrics
        else:
            self.last_score = None
            self.last_metrics = None
        self.reset()
        return self.last_score, self.last_metrics

    def draw(self, frame: np.ndarray) -> None:
        if len(self.points) >= 2:
            pts = np.array(self.points)
            score, metrics = compute_straightness_score(pts)

            # Draw points
            for (cx, cy) in self.points:
                cv2.circle(frame, (int(cx), int(cy)), 3, (255, 0, 0), -1)

            # Draw trajectory path (connected lines)
            for i in range(1, len(self.points)):
                pt1 = (int(self.points[i-1][0]), int(self.points[i-1][1]))
                pt2 = (int(self.points[i][0]), int(self.points[i][1]))
                cv2.line(frame, pt1, pt2, (0, 0, 255), 2)

            # Draw fitted line segment
            p0, direction = fit_line_pca(pts)
            d = direction / (np.linalg.norm(direction) + 1e-12)
            proj_len = np.dot(pts - p0, d)
            min_p = p0 + d * proj_len.min()
            max_p = p0 + d * proj_len.max()

            # Convert to int points for cv2 (x, y)
            p1 = (int(min_p[0]), int(min_p[1]))
            p2 = (int(max_p[0]), int(max_p[1]))
            cv2.line(frame, p1, p2, (0, 255, 255), 2)

            # Draw score
            cv2.putText(frame, f"Straightness: {score:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        elif self.last_score is not None:
            cv2.putText(frame, f"Last Straightness: {self.last_score:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


# Example: instantiate a monitor once before your loop
# monitor = TrajectoryMonitor(min_len=5)

# Example snippets to add inside your existing tracking code:
#  - On successful tracking (inside `if success:` after `x,y,w,h = ...`):
#        cx = int(x + w/2); cy = int(y + h/2)
#        monitor.add_point(cx, cy)
#        monitor.draw(frame)
#  - When tracking fails or you decide the segment ended (where you set
#    tracking_active = False or tracker=None):
#        score, metrics = monitor.finalize()
#        if score is not None:
#            print(f"Segment straightness: {score:.1f}  metrics={metrics}")
#
# The functions are lightweight (pure numpy) and should run fast on CPU.
# Tune parameters: `min_len`, `k` in `compute_straightness_score`, and `max_points` to fit your use case.


# %% [markdown]
# # Open webcam and run tracking

# %%
# Setup Camera
from time import perf_counter


# --- TENNIS BALL COLOR DEFINITION (HSV) ---
# You might need to tune these for your specific lighting!
# "Optic Yellow" is usually around Hue 30-50
LOWER_GREEN = np.array([17, 35, 6])
UPPER_GREEN = np.array([64, 255, 255])

def is_track_good(frame, bbox):
    """
    Verifies if the tracker's bbox likely contains a tennis ball.
    Returns: True if good, False if bad.
    """
    x, y, w, h = [int(v) for v in bbox]


    # COLOR CHECK (The most important one)
    # Extract the image inside the box
    roi = frame[y:y+h, x:x+w]
    if roi.size == 0: return False
    
    # Convert to HSV and create a mask for green/yellow
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, LOWER_GREEN, UPPER_GREEN)
    
    # Count how many pixels are "ball colored"
    ball_pixels = cv2.countNonZero(mask)
    total_pixels = w * h
    
    # If less than 50% of the box is green, we lost it.
    confidence_proxy = ball_pixels / total_pixels
    
    if confidence_proxy < 0.05: 
        return False
        
    return True


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# State variables
tracker = None
tracking_active = False
prev_frame_time = time.perf_counter()

monitor = TrajectoryMonitor(min_len=5)

print("🎾 Tennis Tracker Started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally (mirror effect)
    frame = cv2.flip(frame, 1)

    new_frame_time = time.perf_counter()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    cv2.putText(frame, f"FPS: {fps}", (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # We use a flag to decide if we need to run YOLO this frame
    # By default, if we are tracking, we assume we don't need YOLO yet
    run_yolo = not tracking_active

    # === PHASE 1: TRY TRACKING ===
    if tracking_active:

        start_time = time.perf_counter()

        success, box = tracker.update(frame)
        
        end_time = time.perf_counter()

        print(f"CSRT tracking took {end_time - start_time}ms")

        if success:
            if is_track_good(frame, box):
                # Tracker is happy
                x, y, w, h = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "CSRT TRACKER", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cx = int(x + w/2); 
                cy = int(y + h/2)
                monitor.add_point(cx, cy)
                monitor.draw(frame)
            else:
                # Tracker says "True", but our check says "That's not a ball!"
                print("⚠️ Tracker drifted (Color/Shape mismatch). Resetting...")
                tracking_active = False
                tracker = None
                run_yolo = True 
        else:
            # Tracker FAILED this frame (ball moved too fast or occlusion)
            print("Tracking failed! Switching to YOLO immediate recovery...")
            tracking_active = False
            tracker = None
            run_yolo = True # Force YOLO to run on THIS frame

    # === PHASE 2: SEARCHING (YOLO) ===
    # This runs if we weren't tracking, OR if tracking just failed above
    if run_yolo:

        start_time = time.perf_counter()

        results = yolo_model(frame, verbose=False)

        end_time = time.perf_counter()

        print(f"YOLO took: {end_time - start_time} ms")
        
        best_box = None
        max_conf = 0.0
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Change class_id to 0 if using your custom trained model
                # Change to 32 if using standard YOLOv8n (sports ball)
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Filter for tennis ball (Class 0 usually for custom)
                if class_id == 0 and conf > 0.5:
                    if conf > max_conf:
                        max_conf = conf
                        best_box = box.xyxy[0].cpu().numpy()

        if best_box is not None:
            # Ball found! Initialize tracker for next frame
            x1, y1, x2, y2 = best_box
            w = x2 - x1
            h = y2 - y1
            
            # Create a new tracker instance
            if TRACKER_TYPE == "CSRT":
                tracker = cv2.legacy.TrackerCSRT_create()
            elif TRACKER_TYPE == "MOSSE":
                tracker = cv2.legacy.TrackerMOSSE_create()
            elif TRACKER_TYPE == "KCF":
                tracker = cv2.legacy.TrackerKCF_create()
            else:
                raise IOError("Unrecognized tracker type")  
            
            tracker.init(frame, (int(x1), int(y1), int(w), int(h)))
            tracking_active = True
            
            # Visual feedback for detection
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(frame, f"YOLO DETECT ({max_conf:.2f})", (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cx = int(x1 + w/2); 
            cy = int(y1 + h/2)
            monitor.add_point(cx, cy)
            monitor.draw(frame)
        else:
            # No ball found by YOLO - reset trajectory as ball is out of view
            print("Ball out of view, resetting trajectory...")
            monitor.finalize() # Finalize any current segment
            monitor.reset()    # Clear points for a new trajectory
            cv2.putText(frame, "Searching...", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    cv2.imshow("Tennis Tracker (Auto-Recovery)", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()

# %% [markdown]
# # HSV Upper and Lower bound calibration

# %%
import cv2
import numpy as np

def nothing(x):
    pass

# Create a window
cv2.namedWindow('HSV Tuner')

# Create trackbars for color change
cv2.createTrackbar('H Min', 'HSV Tuner', 17, 179, nothing)
cv2.createTrackbar('S Min', 'HSV Tuner', 35, 255, nothing)
cv2.createTrackbar('V Min', 'HSV Tuner', 6, 255, nothing)
cv2.createTrackbar('H Max', 'HSV Tuner', 64, 179, nothing)
cv2.createTrackbar('S Max', 'HSV Tuner', 255, 255, nothing)
cv2.createTrackbar('V Max', 'HSV Tuner', 255, 255, nothing)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get current positions of trackbars
    hMin = cv2.getTrackbarPos('H Min', 'HSV Tuner')
    sMin = cv2.getTrackbarPos('S Min', 'HSV Tuner')
    vMin = cv2.getTrackbarPos('V Min', 'HSV Tuner')
    hMax = cv2.getTrackbarPos('H Max', 'HSV Tuner')
    sMax = cv2.getTrackbarPos('S Max', 'HSV Tuner')
    vMax = cv2.getTrackbarPos('V Max', 'HSV Tuner')

    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Create Mask
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('HSV Tuner', result)
    
    print(f"LOWER: [{hMin},{sMin},{vMin}]  UPPER: [{hMax},{sMax},{vMax}]", end='\r')
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


