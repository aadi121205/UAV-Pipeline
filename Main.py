from Map import Map  # if saved as mapping.py


# Example input
NW_corner = (28.75375609, 77.11578556)
width_m = 30
length_m = 30

# Get the corners of the square area
corners = Map.make_square_corners(NW_corner, width_m, length_m)

# Create a Map object
mymap = Map(
    square_corners=corners,
    altitude=6,  # meters
    sensor_width_mm=13.2,
    sensor_height_mm=8.8,
    focal_length_mm=8.8,
    image_width_px=4000,
    image_height_px=3000,
    overlap_front=0.7,
    overlap_side=0.7
)

# Get waypoints
waypoints = mymap.get_waypoints()

