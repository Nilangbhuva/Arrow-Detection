import cv2
import numpy as np
import math

# Constants
MATCH_THRESHOLD = 0.8

# Logging functions to trace program execution
def log_initialization():
    print("System initialization...")

def log_template_loaded(template_name):
    print(f"Template {template_name} loaded.")

def log_edge_detection():
    print("Performing edge detection...")

def log_contour_detection():
    print("Detecting contours...")

def log_template_matching():
    print("Performing template matching...")

# Preprocessing: Edge detection and grayscale conversion
def edge_detection(image):
    log_edge_detection()
    edges = cv2.Canny(image, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    return edges

def to_grayscale_and_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    return blurred

# Contour detection
def detect_contours(image):
    log_contour_detection()
    processed = edge_detection(to_grayscale_and_blur(image))
    contours, _ = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Helper functions for arrow detection
def identify_arrow_tip(points, hull_indices):
    for i in range(len(points)):
        if i not in hull_indices:
            next_index = (i + 2) % len(points)
            prev_index = (i - 2 + len(points)) % len(points)
            
            # Check if the points are approximately equal
            distance = np.linalg.norm(points[next_index] - points[prev_index])
            if distance < 1e-3:  # Threshold for considering points equal
                return points[next_index]
    return (-1, -1)

def determine_direction(approx, tip):
    left_points = sum(1 for pt in approx if pt[0][0] < tip[0])
    right_points = sum(1 for pt in approx if pt[0][0] > tip[0])

    if left_points > right_points and left_points > 4:
        return "Left"
    if right_points > left_points and right_points > 4:
        return "Right"
    return "None"

def calculate_angle(p1, p2):
    return math.degrees(math.atan2(p1[1] - p2[1], p1[0] - p2[0]))

# Template matching
def match_and_annotate(frame, template_img, color, label):
    log_template_matching()
    gray_frame = to_grayscale_and_blur(frame)
    best_value = -1
    best_location = (-1, -1)
    best_scale = -1

    for scale in np.arange(0.1, 0.5, 0.027):
        resized_template = cv2.resize(template_img, None, fx=scale, fy=scale)
        result = cv2.matchTemplate(gray_frame, resized_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val > best_value and max_val > MATCH_THRESHOLD:
            best_value = max_val
            best_location = max_loc
            best_scale = scale

    if best_location != (-1, -1):
        w = int(template_img.shape[1] * best_scale)
        h = int(template_img.shape[0] * best_scale)
        top_left = best_location
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(frame, top_left, bottom_right, color, 2)
        
        frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)
        angle = calculate_angle(top_left, frame_center)
        print(f"{label} arrow detected at angle: {angle:.2f}")

# Process each frame to detect arrows
def process_frame(frame):
    contours = detect_contours(frame)
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        hull = cv2.convexHull(approx, returnPoints=False)

        if 4 < len(hull) < 6 and len(hull) + 2 == len(approx) and len(approx) > 6:
            tip = identify_arrow_tip(approx, hull)
            if tip != (-1, -1):
                direction = determine_direction(approx, tip)
                if direction != "None":
                    cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)
                    cv2.circle(frame, tip, 3, (0, 0, 255), -1)
                    print(f"Arrow Direction: {direction}")

# Initialize video capture
def init_video_capture():
    log_initialization()    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
    return cap

# Main function to run the arrow detection
def main():
    log_initialization()
    right_arrow = cv2.imread("Right_Arrow.png", cv2.IMREAD_GRAYSCALE)
    left_arrow = cv2.imread("Left_arrow.png", cv2.IMREAD_GRAYSCALE)
    
    if right_arrow is None or left_arrow is None:
        print("Error loading template images")
        return

    log_template_loaded("Right_Arrow")
    log_template_loaded("Left_Arrow")

    cap = init_video_capture()
    if not cap.isOpened():
        print("Error opening video capture")
        return

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error capturing frame")
            break

        process_frame(frame)
        match_and_annotate(frame, right_arrow, (0, 255, 0), "Right")
        match_and_annotate(frame, left_arrow, (255, 0, 0), "Left")

        cv2.imshow("Video Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
