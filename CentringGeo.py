from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil
import cv2
from ultralytics import YOLO
import time
import numpy as np

# === Connect to drone ===
vehicle = connect("127.0.0.1:14550", wait_ready=True)
print("Connected to drone")

# === Load YOLO model ===
model = YOLO("yolov8n.pt")
print("Model loaded")

# === Open camera feed ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
CENTER_X = FRAME_WIDTH // 2
CENTER_Y = FRAME_HEIGHT // 2
TOLERANCE_X = FRAME_WIDTH * 0.10  # 10% error tolerance
TOLERANCE_Y = FRAME_HEIGHT * 0.10

# === Ensure GUIDED mode ===
if vehicle.mode != VehicleMode("GUIDED"):
    print("Changing mode to GUIDED")
    vehicle.mode = VehicleMode("GUIDED")
    while not vehicle.mode.name == "GUIDED":
        time.sleep(1)
print("Vehicle mode set to GUIDED")

# === Arm and take off ===
if not vehicle.armed:
    print("Arming vehicle")
    vehicle.armed = True
    while not vehicle.armed:
        time.sleep(1)


def takeoff(target_altitude):
    print(f"Taking off to {target_altitude}m")
    vehicle.simple_takeoff(target_altitude)
    while True:
        alt = vehicle.location.global_relative_frame.alt
        if alt >= target_altitude * 0.95:
            print("Reached target altitude")
            break
        time.sleep(1)


takeoff(10)


# === Send local NED velocity ===
def send_ned_velocity(vx, vy, vz, duration=1):
    """
    vx: forward/backward (m/s)
    vy: left/right (m/s)
    vz: up/down (m/s) – we keep it 0 here
    """
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,
        0,
        0,  # time_boot_ms, target_system, target_component
        mavutil.mavlink.MAV_FRAME_BODY_NED,  # relative to drone body
        0b0000111111000111,  # velocity enabled
        0,
        0,
        0,  # x, y, z positions
        vx,
        vy,
        vz,  # velocities in m/s
        0,
        0,
        0,  # accelerations
        0,
        0,
    )
    for _ in range(duration):
        vehicle.send_mavlink(msg)
        time.sleep(1)


# === Main control loop ===
print("Tracking...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.5)
    boxes = results[0].boxes

    person_found = False

    for box in boxes:
        cls = int(box.cls[0])
        if model.names[cls] == "person":
            person_found = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox_center_x = (x1 + x2) // 2
            bbox_center_y = (y1 + y2) // 2

            offset_x = bbox_center_x - CENTER_X
            offset_y = bbox_center_y - CENTER_Y

            # Draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.line(frame, (CENTER_X, 0), (CENTER_X, FRAME_HEIGHT), (255, 0, 0), 2)
            cv2.line(frame, (0, CENTER_Y), (FRAME_WIDTH, CENTER_Y), (255, 0, 0), 2)
            cv2.circle(frame, (bbox_center_x, bbox_center_y), 5, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"Offset X: {offset_x} Y: {offset_y}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

            # === Movement Control Logic ===
            if abs(offset_x) > TOLERANCE_X or abs(offset_y) > TOLERANCE_Y:
                # Proportional control to velocity (scale factor)
                scale = 0.0025  # You can tune this
                vy = -scale * offset_x  # Left/right
                vx = -scale * offset_y  # Forward/backward (inverted Y axis)
                vx = np.clip(vx, -0.5, 0.5)
                vy = np.clip(vy, -0.5, 0.5)
                send_ned_velocity(vx, vy, 0, duration=1)

            break

    if not person_found:
        cv2.putText(
            frame,
            "No person detected",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    cv2.imshow("Drone Cam", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
vehicle.close()
