import cv2
import cvzone
import numpy as np
import pickle
import requests

# Initialize the live camera feed
LIVE_CAMERA_URL = "http://192.168.236.227:8080/shot.jpg"

def get_live_frame():
    try:
        response = requests.get(LIVE_CAMERA_URL, timeout=5)
        video = np.array(bytearray(response.content), dtype=np.uint8)
        frame = cv2.imdecode(video, -1)
        return frame
    except Exception as e:
        print(f"Error fetching live camera feed: {e}")
        return None

drawing = False
area_names = []

# Load previous polylines and area names
try:
    with open("manish", "rb") as f:
        data = pickle.load(f)
        polylines, area_names = data['polylines'], data['area_names']
except:
    polylines = []

current_names = ""
points = []
currvalue = 0  # Initialize currvalue as a global variable

def draw(event, x, y, flags, param):
    global points, drawing, currvalue  # Reference global variables
    if event == cv2.EVENT_LBUTTONDOWN:
        if currvalue < 4:
            if not drawing:
                drawing = True
                print(x, y)
                points = [(x, y)]
                currvalue += 1
            else:
                print(x, y)
                currvalue += 1
                points.append((x, y))
        else:
            drawing = False
            currvalue = 0
            current_name = input('Area name: ')
            if current_name:
                area_names.append(current_name)
                polylines.append(np.array(points, np.int32))

def draw_rectangle_with_points(image, points, color=(0, 255, 0), thickness=2):
    """
    Draws a rectangle using 4 points on the given image.

    :param image: The image on which to draw the rectangle
    :param points: List of 4 points (x, y) defining the rectangle
    :param color: Color of the rectangle (default is green)
    :param thickness: Thickness of the rectangle lines
    """
    # Ensure points are in proper order (e.g., top-left, top-right, bottom-right, bottom-left)
    points = np.array(points, dtype=np.int32)

    # Draw the lines connecting the points
    for i in range(len(points)):
        cv2.line(image, tuple(points[i]), tuple(points[(i + 1) % len(points)]), color, thickness)

while True:
    frame = get_live_frame()

    if frame is None:
        print("Retrying to fetch live frame...")
        continue

    frame = cv2.resize(frame, (1020, 500))

    for i, polyline in enumerate(polylines):
        # cv2.polylines(frame, [polyline], True, (0, 0, 255), 2)
        rectangle_points = polyline
        draw_rectangle_with_points(frame, rectangle_points)
        cvzone.putTextRect(frame, f'{area_names[i]}', tuple(polyline[0]), 1, 1)

    cv2.imshow('FRAME', frame)
    cv2.setMouseCallback('FRAME', draw)

    key = cv2.waitKey(100) & 0xFF

    # Save polylines and area names to file
    if key == ord('s'):
        with open("manish", "wb") as f:
            data = {'polylines': polylines, 'area_names': area_names}
            pickle.dump(data, f)

    # Exit the program
    if key == ord('q'):
        break

cv2.destroyAllWindows()
