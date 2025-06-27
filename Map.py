import numpy as np
from geographiclib.geodesic import Geodesic
import matplotlib.pyplot as plt

def plot_waypoints(square_corners, waypoints):
    # Extract lats and lons
    lats = [wp[0] for wp in waypoints]
    lons = [wp[1] for wp in waypoints]
    # Corner polygon
    corner_lats = [pt[0] for pt in square_corners] + [square_corners[0][0]]
    corner_lons = [pt[1] for pt in square_corners] + [square_corners[0][1]]

    plt.figure(figsize=(8, 8))
    # Draw area
    plt.plot(corner_lons, corner_lats, 'k-', label='Mapping Area', linewidth=2)
    # Draw waypoints
    plt.scatter(lons, lats, c='red', s=15, label='Waypoints')
    # Draw lines (flight path)
    plt.plot(lons, lats, 'b--', alpha=0.5, label='Path')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Mapping Mission Waypoints')
    plt.legend()
    plt.grid(True)
    plt.show()

def gps_to_xy(ref_lat, ref_lon, lat, lon):
    """Convert lat/lon to local tangent XY (meters) using the reference point as (0,0)."""
    g = Geodesic.WGS84.Inverse(ref_lat, ref_lon, lat, lon)
    az = np.radians(g['azi1'])
    x = g['s12'] * np.sin(az)
    y = g['s12'] * np.cos(az)
    return x, y

def xy_to_gps(ref_lat, ref_lon, x, y):
    """Convert local tangent XY (meters) to lat/lon using the reference point as (0,0)."""
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
    # Reference point (first corner)
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
            lat, lon = xy_to_gps(ref_lat, ref_lon, x, y)
            row_points.append((lat, lon, altitude))
            x += step_x
        if row % 2 == 1:
            row_points = row_points[::-1]
        waypoints.extend(row_points)
        y += step_y
        row += 1
    return waypoints

def make_square_corners(NW, width_m, length_m):
    """
    Given NW corner, width (East, in meters), and length (South, in meters), 
    compute the other three corners using geodesic calculation for accuracy.
    Returns corners as [NW, NE, SE, SW] (clockwise).
    """
    NW_lat, NW_lon = NW

    # Move east for width (azimuth 90)
    NE = Geodesic.WGS84.Direct(NW_lat, NW_lon, 90, width_m)
    NE_lat, NE_lon = NE['lat2'], NE['lon2']

    # Move south from NE for length (azimuth 180)
    SE = Geodesic.WGS84.Direct(NE_lat, NE_lon, 180, length_m)
    SE_lat, SE_lon = SE['lat2'], SE['lon2']

    # Move south from NW for length (azimuth 180)
    SW = Geodesic.WGS84.Direct(NW_lat, NW_lon, 180, length_m)
    SW_lat, SW_lon = SW['lat2'], SW['lon2']

    return [(NW_lat, NW_lon), (NE_lat, NE_lon), (SE_lat, SE_lon), (SW_lat, SW_lon)]  # Clockwise

# ====== Example usage ======
if __name__ == "__main__":
    # North-West corner (lat, lon)
    NW = (28.75375609, 77.11578556)
    width_m = 30  # in meters
    length_m = 30  # in meters

    square = make_square_corners(NW, width_m, length_m)
    print("Square Corners (lat, lon):")
    for corner in square:
        print(corner)
    print("\nGenerated Waypoints:")
    waypoints = generate_mapping_waypoints(
        square_corners=square,
        altitude=6,  # meters
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

    plot_waypoints(square, waypoints)
