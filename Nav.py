import time
import threading
from dronekit import connect, VehicleMode
from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    a = (
        sin((lat2 - lat1) / 2) ** 2
        + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    )
    c = 2 * asin(sqrt(a))
    return c * 6371 * 1000

def Caluate_best_misssion(a, b, c, d):
    """
    Claculate the best mission based on square made by the above points in the order of a, b, c, d. To execue a maping mission, the drone will fly in a square pattern.
    """
    distance_ab = haversine(a[1], a[0], b[1], b[0])
    distance_bc = haversine(b[1], b[0], c[1], c[0])
    distance_cd = haversine(c[1], c[0], d[1], d[0])
    distance_da = haversine(d[1], d[0], a[1], a[0])
    
    total_distance = distance_ab + distance_bc + distance_cd + distance_da
    return total_distance

class Telem:
    def __init__(self):
        self.DroneIP = "udp:0.0.0.0:14550"  # Change to your UAV's IP
        self.vehicle = None
        self.connect_uav()
        threading.Thread(target=self.send_telemetry_data, daemon=True).start()

    def connect_uav(self):
        try:
            print("Trying to connect UAV...")
            self.vehicle = connect(self.DroneIP, wait_ready=True)
            print("[UAV] Connected to UAV")
        except Exception as e:
            print("[UAV] Connection error: ", str(e))
            time.sleep(1)

    def send_telemetry_data(self):
        while True:
            try:
                v = self.vehicle
                telemetry = {
                    "mode": v.mode.name,
                    "armed": v.armed,
                    "location": {
                        "lat": v.location.global_frame.lat,
                        "lon": v.location.global_frame.lon,
                        "alt": v.location.global_frame.alt,
                    },
                    "battery": v.battery.level,
                    "gps": {
                        "fix_type": v.gps_0.fix_type,
                        "satellites_visible": v.gps_0.satellites_visible,
                    },
                    "heading": v.heading,
                    "velocity": v.velocity,
                    "last_heartbeat": v.last_heartbeat,
                }
                return telemetry
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
            print(f" Altitude: {self.vehicle.location.global_relative_frame.alt}")
            if self.vehicle.location.global_relative_frame.alt >= target_altitude * 0.95:
                print("Reached target altitude")
                break
            time.sleep(1)


if __name__ == "__main__":
    telem = Telem()
    while True:
        time.sleep(0.5)
        telemetry_data = telem.send_telemetry_data()
        if telemetry_data:
            print("Telemetry Data:", telemetry_data)

