import numpy as np
from geographiclib.geodesic import Geodesic

def gps_to_xy(ref_lat, ref_lon, lat, lon):
    """Convert lat/lon to local tangent XY (meters) using the reference point as (0,0)."""
    g = Geodesic.WGS84.Inverse(ref_lat, ref_lon, lat, lon)
    az = np.radians(g['azi1'])
    x = g['s12'] * np.sin(az)
    y = g['s12'] * np.cos(az)
    return x, y

def xy_to_gps(ref_lat, ref_lon, x, y):
    """Convert local tangent XY (meters) to lat/lon using the reference point as (0,0)."""
    # Distance and azimuth from origin
    s = np.hypot(x, y)
    azi = np.degrees(np.arctan2(x, y))
    g = Geodesic.WGS84.Direct(ref_lat, ref_lon, azi, s)
    return g['lat2'], g['lon2']

def calculate_camera_footprint(altitude, sensor_width_mm, sensor_height_mm, focal_length_mm, image_width_px, image_height_px):
    """Calculate ground footprint of camera at given altitude."""
    GSD_x = (sensor_width_mm * altitude) / (focal_length_mm * image_width_px)  # meters per pixel
    GSD_y = (sensor_height_mm * altitude) / (focal_length_mm * image_height_px)
    footprint_x = GSD_x * image_width_px
    footprint_y = GSD_y * image_height_px
    return footprint_x, footprint_y

def generate_mapping_waypoints(square_corners, altitude, sensor_width_mm, sensor_height_mm, focal_length_mm,
                              image_width_px, image_height_px, overlap_front=0.7, overlap_side=0.7):
    """
    square_corners: list of 4 (lat, lon) tuples, in order
    altitude: mapping altitude in meters
    camera/sensor parameters as per above
    """
    # Reference point (lower left)
    ref_lat, ref_lon = square_corners[0]
    # Convert all corners to XY
    xy_corners = [gps_to_xy(ref_lat, ref_lon, lat, lon) for lat, lon in square_corners]
    # Fit axis-aligned bounding box
    xs, ys = zip(*xy_corners)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    # Camera footprint
    footprint_x, footprint_y = calculate_camera_footprint(
        altitude, sensor_width_mm, sensor_height_mm, focal_length_mm, image_width_px, image_height_px
    )
    step_x = footprint_x * (1 - overlap_side)
    step_y = footprint_y * (1 - overlap_front)
    # Build grid
    waypoints = []
    y = min_y
    row = 0
    while y <= max_y:
        row_points = []
        x = min_x
        while x <= max_x:
            # Check if within polygon (optional)
            lat, lon = xy_to_gps(ref_lat, ref_lon, x, y)
            row_points.append((lat, lon, altitude))
            x += step_x
        # Zig-zag path for efficiency
        if row % 2 == 1:
            row_points = row_points[::-1]
        waypoints.extend(row_points)
        y += step_y
        row += 1
    return waypoints

# ====== Example usage ======
if __name__ == "__main__":
    # (Corners in counterclockwise or clockwise order)
    square = [
        (28.7500, 77.1100),  # SW
        (28.7500, 77.1200),  # SE
        (28.7600, 77.1200),  # NE
        (28.7600, 77.1100)   # NW
    ]
    waypoints = generate_mapping_waypoints(
        square_corners=square,
        altitude=100,  # meters
        sensor_width_mm=13.2,
        sensor_height_mm=8.8,
        focal_length_mm=8.8,
        image_width_px=4000,
        image_height_px=3000,
        overlap_front=0.7,
        overlap_side=0.7
    )
    for wp in waypoints:
        print(wp)
