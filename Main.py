from dronekit import connect, VehicleMode
import cv2
from ultralytics import YOLO
import time
import math

# === Connect to drone ===
vehicle = connect('127.0.0.1:14550', wait_ready=True)  # Replace with your telemetry port

# === Load YOLO model ===
model = YOLO('yolov8n.pt')  # Or yolov5s.pt if using YOLOv5

# === Open camera feed ===
cap = cv2.VideoCapture(0)  # Use 0 or your video stream path

FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
CENTER_X = FRAME_WIDTH // 2
CENTER_Y = FRAME_HEIGHT // 2

# === PID control or simple offset logic ===
def adjust_yaw_based_on_offset(offset_x):
    yaw_rate = 0.1  # Yaw adjustment speed
    threshold = 40  # Pixels
    if abs(offset_x) < threshold:
        return  # Already centered

    direction = -1 if offset_x > 0 else 1  # Right = Negative Yaw, Left = Positive
    condition_yaw(direction * 5)

# === Yaw control ===
def condition_yaw(relative_yaw):
    """
    relative_yaw: in degrees, positive for clockwise
    """
    msg = vehicle.message_factory.command_long_encode(
        0, 0,  # target_system, target_component
        mavutil.mavlink.MAV_CMD_CONDITION_YAW,
        0,
        abs(relative_yaw),  # target angle
        10,                 # yaw speed deg/s
        1 if relative_yaw >= 0 else -1,  # direction
        1, 0, 0, 0)          # relative offset
    vehicle.send_mavlink(msg)

# === Main loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.5)
    if results and len(results[0].boxes):
        for box in results[0].boxes:
            cls = int(box.cls[0])
            if model.names[cls] == 'person':
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_center_x = (x1 + x2) // 2
                offset_x = person_center_x - CENTER_X

                adjust_yaw_based_on_offset(offset_x)

                # Draw
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.line(frame, (CENTER_X, 0), (CENTER_X, FRAME_HEIGHT), (255,0,0), 1)
                cv2.putText(frame, f'Offset: {offset_x}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                break

    cv2.imshow('Drone Cam', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
vehicle.close()
