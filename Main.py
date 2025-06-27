import threading
import queue
import time
from dronekit import connect, VehicleMode, LocationGlobalRelative
import cv2
import torch

# --- CONFIGURATION ---
CAMERA_ID = 0  # Change if needed
YOLO_MODEL_PATH = "yolov5s.pt"  # Change to your trained model
PATH = "Video.mp4"  # Path to video file if using video instead of camera

# --- THREAD-SAFE QUEUE ---
frame_queue = queue.Queue(maxsize=10)  # For camera frames

# --- DRONEKIT: Connect to vehicle ---
vehicle = connect('udp:127.0.0.1:14550', wait_ready=True)  # Adjust for your setup

def arm_and_takeoff(target_altitude):
    print("Arming motors")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True
    while not vehicle.armed:
        print(" Waiting for arming...")
        time.sleep(1)
    print("Taking off!")
    vehicle.simple_takeoff(target_altitude)
    while True:
        print(f" Altitude: {vehicle.location.global_relative_frame.alt}")
        if vehicle.location.global_relative_frame.alt >= target_altitude * 0.95:
            print("Reached target altitude")
            break
        time.sleep(1)

def navigation_thread():
    """Controls drone mission (example: takeoff, move to waypoints)."""
    arm_and_takeoff(10)
    # Example: Go to a waypoint
    point1 = LocationGlobalRelative(28.7536, 77.1154, 10)
    vehicle.simple_goto(point1)
    time.sleep(30)  # Loiter for demo
    vehicle.mode = VehicleMode("LAND")
    print("Landing")
    time.sleep(10)
    print("Navigation done")

def camera_feed_thread():
    """Reads frames from camera and puts into queue."""
    cap = cv2.VideoCapture(CAMERA_ID)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        if not frame_queue.full():
            frame_queue.put(frame)
        time.sleep(0.03)  # ~30 fps

def yolo_inference_thread():
    """Loads YOLO, reads frames from queue, runs inference."""
    # Load model (PyTorch Hub)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=YOLO_MODEL_PATH, force_reload=True)
    model.eval()
    print("YOLO model loaded")
    while True:
        frame = frame_queue.get()  # Waits for a frame
        # Inference
        results = model(frame)
        print(results.pandas().xyxy[0])  # Print results table (you can process detections)
        # Optional: show frame with detections
        # results.show()
        # Or do something based on detections

# --- START THREADS ---
threads = []
t_nav = threading.Thread(target=navigation_thread, daemon=True)
t_cam = threading.Thread(target=camera_feed_thread, daemon=True)
t_yolo = threading.Thread(target=yolo_inference_thread, daemon=True)
threads.extend([t_nav, t_cam, t_yolo])

for t in threads:
    t.start()

# --- MAIN LOOP (keep alive) ---
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Exiting")
    vehicle.close()
