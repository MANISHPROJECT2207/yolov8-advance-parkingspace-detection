import cv2
import numpy as np
import cvzone
import pickle
import requests
import pandas as pd
from ultralytics import YOLO
import easyocr  # For OCR

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


def draw_rectangle_with_points(image, points, color, thickness=2):
    """
    Draws a rectangle using 4 points on the given image.

    :param image: The image on which to draw the rectangle
    :param points: List of 4 points (x, y) defining the rectangle
    :param color: Color of the rectangle (default is green)
    :param thickness: Thickness of the rectangle lines
    """
    points = np.array(points, dtype=np.int32)
    for i in range(len(points)):
        if color == 1:
            cv2.line(image, tuple(points[i]), tuple(points[(i + 1) % len(points)]), (0, 255, 0), thickness)
        else:
            cv2.line(image, tuple(points[i]), tuple(points[(i + 1) % len(points)]), (0, 0, 255), thickness)

# Load YOLO model
model = YOLO('yolov8s.pt')

# Load previous polylines and area names
try:
    with open("manish", "rb") as f:
        data = pickle.load(f)
        polylines, area_names = data['polylines'], data['area_names']
except:
    polylines = []
    area_names = []

# Load class names
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def detect_cars_and_number_plates(frame):
    results = model.predict(frame)
    detections = results[0].boxes.data
    detections_df = pd.DataFrame(detections).astype("float")

    car_centroids = []
    car_boxes = []
    number_plates = []

    for _, row in detections_df.iterrows():
        x1, y1, x2, y2, _, class_id = map(int, row)
        class_name = class_list[class_id]
        if 'car' in class_name:
            # Car detection
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            car_centroids.append((cx, cy))
            car_boxes.append((x1, y1, x2, y2))

            # Extract potential number plate region (cropping)
            car_crop = frame[y1:y2, x1:x2]
            results = reader.readtext(car_crop)
            if results:
                # Take the first detected text from EasyOCR
                number_plate = results[0][1]
                number_plates.append((number_plate, (x1, y1)))

    return car_centroids, number_plates

while True:
    frame = get_live_frame()

    if frame is None:
        print("Retrying to fetch live frame...")
        continue

    frame = cv2.resize(frame, (1020, 500))
    car_centroids, number_plates = detect_cars_and_number_plates(frame)

    free_space_counter = []
    for i, polyline in enumerate(polylines):
        rectangle_points = polyline
        color = 1
        draw_rectangle_with_points(frame, rectangle_points, color)
        cvzone.putTextRect(frame, f'{area_names[i]}', tuple(polyline[0]), 1, 1)
        for cx, cy in car_centroids:
            result = cv2.pointPolygonTest(polyline, (cx, cy), False)
            if result >= 0:
                rectangle_points = polyline
                color = 2
                draw_rectangle_with_points(frame, rectangle_points, color)
                free_space_counter.append(tuple(map(tuple, polyline)))  # Convert to hashable type

    car_count = len(set(free_space_counter))  # Works now
    free_space = len(polylines) - car_count

    # Display car count and free space
    cvzone.putTextRect(frame, f'CAR COUNT: {car_count}', (50, 60), 2, 2)
    cvzone.putTextRect(frame, f'FREE SPACE: {free_space}', (50, 100), 2, 2)

    # Display detected number plates
    for number_plate, (x, y) in number_plates:
        cvzone.putTextRect(frame, f'{number_plate}', (x, y - 10), 1, 1, colorR=(0, 255, 0))

    cv2.imshow('FRAME', frame)
    key = cv2.waitKey(1) & 0xFF

    # Exit the program
    if key == ord('q'):
        break

cv2.destroyAllWindows()