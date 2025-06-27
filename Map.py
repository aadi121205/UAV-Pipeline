import numpy as np
from geographiclib.geodesic import Geodesic

class Map:
    def __init__(self, square_corners, altitude, sensor_width_mm, sensor_height_mm,
                 focal_length_mm, image_width_px, image_height_px, overlap_front=0.7, overlap_side=0.7):
        self.square_corners = square_corners
        self.altitude = altitude
        self.sensor_width_mm = sensor_width_mm
        self.sensor_height_mm = sensor_height_mm
        self.focal_length_mm = focal_length_mm
        self.image_width_px = image_width_px
        self.image_height_px = image_height_px
        self.overlap_front = overlap_front
        self.overlap_side = overlap_side

    @staticmethod
    def gps_to_xy(ref_lat, ref_lon, lat, lon):
        g = Geodesic.WGS84.Inverse(ref_lat, ref_lon, lat, lon)
        az = np.radians(g['azi1'])
        x = g['s12'] * np.sin(az)
        y = g['s12'] * np.cos(az)
        return x, y

    @staticmethod
    def xy_to_gps(ref_lat, ref_lon, x, y):
        s = np.hypot(x, y)
        azi = np.degrees(np.arctan2(x, y))
        g = Geodesic.WGS84.Direct(ref_lat, ref_lon, azi, s)
        return g['lat2'], g['lon2']

    @staticmethod
    def calculate_camera_footprint(altitude, sensor_width_mm, sensor_height_mm, focal_length_mm, image_width_px, image_height_px):
        GSD_x = (sensor_width_mm * altitude) / (focal_length_mm * image_width_px)  # meters per pixel
        GSD_y = (sensor_height_mm * altitude) / (focal_length_mm * image_height_px)
        footprint_x = GSD_x * image_width_px
        footprint_y = GSD_y * image_height_px
        return footprint_x, footprint_y

    def get_waypoints(self):
        ref_lat, ref_lon = self.square_corners[0]
        xy_corners = [self.gps_to_xy(ref_lat, ref_lon, lat, lon) for lat, lon in self.square_corners]
        xs, ys = zip(*xy_corners)
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        footprint_x, footprint_y = self.calculate_camera_footprint(
            self.altitude, self.sensor_width_mm, self.sensor_height_mm, self.focal_length_mm,
            self.image_width_px, self.image_height_px
        )
        step_x = footprint_x * (1 - self.overlap_side)
        step_y = footprint_y * (1 - self.overlap_front)

        waypoints = []
        y = min_y
        row = 0
        while y <= max_y:
            row_points = []
            x = min_x
            while x <= max_x:
                lat, lon = self.xy_to_gps(ref_lat, ref_lon, x, y)
                row_points.append((lat, lon, self.altitude))
                x += step_x
            if row % 2 == 1:
                row_points = row_points[::-1]
            waypoints.extend(row_points)
            y += step_y
            row += 1
        return waypoints

    @staticmethod
    def make_square_corners(NW, width_m, length_m):
        NW_lat, NW_lon = NW
        NE = Geodesic.WGS84.Direct(NW_lat, NW_lon, 90, width_m)
        NE_lat, NE_lon = NE['lat2'], NE['lon2']
        SE = Geodesic.WGS84.Direct(NE_lat, NE_lon, 180, length_m)
        SE_lat, SE_lon = SE['lat2'], SE['lon2']
        SW = Geodesic.WGS84.Direct(NW_lat, NW_lon, 180, length_m)
        SW_lat, SW_lon = SW['lat2'], SW['lon2']
        return [(NW_lat, NW_lon), (NE_lat, NE_lon), (SE_lat, SE_lon), (SW_lat, SW_lon)]
