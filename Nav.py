import time
import threading
from dronekit import connect, VehicleMode, LocationGlobalRelative
from math import radians, cos, sin, asin, sqrt
from Map import Map  # Ensure your Map class provides needed staticmethods
import os
import cv2
from ultralytics import YOLO
from pymavlink import mavutil
import numpy as np

OutputDirectory = os.path.join(os.getcwd(), "Output")
if not os.path.exists(OutputDirectory):
    os.makedirs(OutputDirectory)

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

cap.release()

# === Send local NED velocity ===

def send_ned_velocity(vehicle, vx, vy, vz, duration=1):
    """
    Move vehicle in direction based on specified velocity vectors.
    Args:
        vehicle: dronekit.Vehicle object
        vx: velocity in x direction (m/s), forward is positive
        vy: velocity in y direction (m/s), right is positive
        vz: velocity in z direction (m/s), down is positive
        duration: seconds to send the command (sends at 1Hz)
    """
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,      # time_boot_ms (ignored)
        0, 0,   # target system, target component
        mavutil.mavlink.MAV_FRAME_BODY_NED,  # frame
        0b0000111111000111,  # type_mask (only velocity enabled)
        0, 0, 0,             # x, y, z positions (not used)
        vx, vy, vz,          # velocities (m/s)
        0, 0, 0,             # accelerations (not used)
        0, 0                 # yaw, yaw_rate (not used)
    )
    for _ in range(duration):
        vehicle.send_mavlink(msg)
        time.sleep(1)

def Geo():
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

def CaptureImage(OutputDirectory=OutputDirectory, image_index=0):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return None
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from video device.")
        return None
    filename = os.path.join(
        OutputDirectory, f"image_{int(time.time())}_{image_index}.jpg"
    )
    cv2.imwrite(filename, frame)
    cap.release()
    print(f"Image captured and saved to {filename}")
    return filename


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    a = (
        sin((lat2 - lat1) / 2) ** 2
        + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    )
    c = 2 * asin(sqrt(a))
    return c * 6371 * 1000  # meters


def Caluate_best_misssion(NW=(28.75375609, 77.11578556), Altitude=6):
    # Example input
    NW_corner = NW
    width_m = 30  # meters
    length_m = 30
    # Use your Map's staticmethod to make corners
    corners = Map.make_square_corners(NW_corner, width_m, length_m)
    mymap = Map(
        square_corners=corners,
        altitude=Altitude,  # meters
        sensor_width_mm=6.4,
        sensor_height_mm=4.0,
        focal_length_mm=1.93,  # use RGB value
        image_width_px=1280,  # RGB maximum resolution
        image_height_px=800,  # RGB maximum resolution
        overlap_front=0.5,
        overlap_side=0.5,
    )
    waypoints = mymap.get_waypoints()
    return waypoints


class Telem:
    def __init__(self):
        self.DroneIP = "udp:0.0.0.0:14550"  # Change as needed
        self.vehicle = None
        self.connect_uav()
        time.sleep(2)
        threading.Thread(target=self.send_telemetry_data, daemon=True).start()
        self.execute_maping_mission()
        self.close()

    def connect_uav(self):
        try:
            print("Trying to connect UAV...")
            self.vehicle = connect(self.DroneIP, wait_ready=True)
            print("[UAV] Connected to UAV")
        except Exception as e:
            print("[UAV] Connection error: ", str(e))
            time.sleep(2)
            # Optionally retry connect here

    def send_telemetry_data(self):
        while True:
            try:
                v = self.vehicle
                telemetry = {
                    "mode": v.mode.name if v else None,
                    "armed": v.armed if v else None,
                    "location": {
                        "lat": v.location.global_frame.lat if v else None,
                        "lon": v.location.global_frame.lon if v else None,
                        "alt": v.location.global_frame.alt if v else None,
                    },
                    "battery": v.battery.level if v else None,
                    "gps": {
                        "fix_type": v.gps_0.fix_type if v else None,
                        "satellites_visible": v.gps_0.satellites_visible if v else None,
                    },
                    "heading": v.heading if v else None,
                    "velocity": v.velocity if v else None,
                    "last_heartbeat": v.last_heartbeat if v else None,
                }
                # print("TELEMETRY:", telemetry)
            except Exception as e:
                print("[UAV] Error receiving telemetry data:", str(e))
            time.sleep(1)

    def arm_and_takeoff(self, target_altitude):
        print("Arming motors")
        self.vehicle.mode = VehicleMode("GUIDED")
        self.vehicle.armed = True
        while not self.vehicle.armed:
            print(" Waiting for arming...")
            time.sleep(1)
        print("Taking off!")
        self.vehicle.simple_takeoff(target_altitude)
        while True:
            alt = self.vehicle.location.global_relative_frame.alt
            print(f" Altitude: {alt:.2f}")
            if alt >= target_altitude * 0.95:
                print("Reached target altitude")
                break
            time.sleep(1)

    def execute_maping_mission(self):
        waypoints = Caluate_best_misssion()
        if not waypoints or not isinstance(waypoints, list):
            print("No waypoints generated.")
            return
        if not self.vehicle or not self.vehicle.armed:
            print("Vehicle not connected or not armed.")
            self.arm_and_takeoff(10)
        else:
            print("Already armed.")
        print("Starting mapping mission...")
        for i, waypoint in enumerate(waypoints):
            lat, lon, alt = waypoint
            print(f"Moving to waypoint {i+1}: ({lat}, {lon}, {alt})")
            loc = LocationGlobalRelative(lat, lon, alt)
            self.vehicle.simple_goto(loc)
            while True:
                current_location = self.vehicle.location.global_relative_frame
                distance = haversine(
                    current_location.lon, current_location.lat, lon, lat
                )
                print(f"Distance to waypoint {i+1}: {distance:.2f} meters")
                if distance < 3:
                    print(f"Reached waypoint {i+1}")
                    # Capture image at this waypoint
                    image_path = CaptureImage(OutputDirectory, i + 1)
                    time.sleep(2)  # Wait for a bit before next waypoint
                    break
                time.sleep(1)

    def close(self):
        print("RTL")
        if self.vehicle and self.vehicle.mode != VehicleMode("RTL"):
            print("Setting mode to RTL (Return to Launch).")
            self.vehicle.mode = VehicleMode("RTL")
            while self.vehicle.mode != VehicleMode("RTL"):
                print(" Waiting for mode change...")
                time.sleep(1)
        time.sleep(15)
        if self.vehicle and self.vehicle.armed:
            print("Disarming vehicle.")
            self.vehicle.armed = False
            while self.vehicle.armed:
                print(" Waiting for disarming...")
                time.sleep(1)
        if self.vehicle:
            print("Closing vehicle connection.")
            self.vehicle.close()


if __name__ == "__main__":
    telem = Telem()
