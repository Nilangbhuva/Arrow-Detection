import cv2
import numpy as np

# Add text to the image
def add_text(img, text, position, font_size=1, color=(255, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, position, font, font_size, color, 2, cv2.LINE_AA)

def display_images():
    # Step 1: Capture an image
    cap = cv2.VideoCapture(0)
    ret, original_image = cap.read()
    cap.release()

    if not ret:
        print("Failed to capture image")
        return

    # Step 2: Resize to 1:3 ratio
    height, width = original_image.shape[:2]
    new_width = width
    new_height = width // 3
    resized_image = cv2.resize(original_image, (new_width, new_height))

    # Match heights by padding the resized image
    padding_top = (height - new_height) // 2
    padding_bottom = height - new_height - padding_top
    resized_image_padded = cv2.copyMakeBorder(
        resized_image, padding_top, padding_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    # Step 3: Add details to the panorama
    gps_data = {
        "latitude": 37.7749,
        "longitude": -122.4194,
        "altitude": 15.7,  # Example altitude
        "accuracy": 3.5,   # Example accuracy
    }
    scale = "Scale: 1:3"
    directions = "Cardinal Directions: N | S | E | W"
    gps_info = f"GPS: {gps_data['latitude']:.5f}, {gps_data['longitude']:.5f}"
    elevation = f"Elevation: {gps_data['altitude']:.2f} m"
    accuracy = f"Accuracy: ±{gps_data['accuracy']:.2f} m"

    overlay = resized_image_padded.copy()
    add_text(overlay, scale, (10, 30))
    add_text(overlay, directions, (10, 70))
    add_text(overlay, gps_info, (10, 110))
    add_text(overlay, elevation, (10, 150))
    add_text(overlay, accuracy, (10, 190))

    # Combine text and panorama image
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, resized_image_padded, 1 - alpha, 0, resized_image_padded)

    # Step 4: Display both images side by side
    combined = np.hstack((original_image, resized_image_padded))
    cv2.imshow("Original (Left) and Panorama (Right)", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    display_images()


# import cv2
# import numpy as np
# from PIL import ImageFont, ImageDraw, Image
# # import gpsd  # Use gpsd for getting GPS data; make sure it's installed
# # import geopy.distance

# # def get_gps_data():
# #     gpsd.connect()  # Connect to the GPS daemon
# #     packet = gpsd.get_current()
# #     return {
# #         "latitude": packet.lat,
# #         "longitude": packet.lon,
# #         "altitude": packet.alt,
# #         "accuracy": packet.hdop,
# #     }

# def add_text(img, text, position, font_size=1, color=(255, 255, 255)):
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     cv2.putText(img, text, position, font, font_size, color, 2, cv2.LINE_AA)

# def display_images():
#     # Step 1: Capture an image
#     cap = cv2.VideoCapture(0)
#     ret, original_image = cap.read()
#     cap.release()

#     if not ret:
#         print("Failed to capture image")
#         return

#     # Step 2: Resize to 1:3 ratio
#     height, width = original_image.shape[:2]
#     new_width = width
#     new_height = width // 3
#     resized_image = cv2.resize(original_image, (new_width, new_height))

#     # Match heights by padding the resized image
#     padding_top = (height - new_height) // 2
#     padding_bottom = height - new_height - padding_top
#     resized_image_padded = cv2.copyMakeBorder(
#         resized_image, padding_top, padding_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0]
#     )
    
#     # Step 3: Add details to panorama
#     # gps_data = get_gps_data()
#     gps_data = {
#         "latitude": 37.7749,
#         "longitude": -122.4194,
#         "altitude": 0,
#         "accuracy": 0,}
#     scale = "Scale: 1:3"
#     # directions = "Cardinal Directions: N | S | E | W"
#     # gps_info = f"GPS: {gps_data['latitude']:.5f}, {gps_data['longitude']:.5f}"
#     # elevation = f"Elevation: {gps_data['altitude']:.2f} m"
#     # accuracy = f"Accuracy: {gps_data['accuracy']:.2f} m"
    

#     overlay = resized_image.copy()
#     add_text(overlay, scale, (10, 30))
#     add_text(overlay, directions, (10, 70))
#     add_text(overlay, gps_info, (10, 110))
#     add_text(overlay, elevation, (10, 150))
#     add_text(overlay, accuracy, (10, 190))

#     # Combine text and panorama image
#     alpha = 0.7
#     cv2.addWeighted(overlay, alpha, resized_image, 1 - alpha, 0, resized_image)

#     # Step 4: Display both images side by side
#     combined = np.hstack((original_image, resized_image))
#     cv2.imshow("Original (Left) and Panorama (Right)", combined)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     display_images()
