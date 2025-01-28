from ultralytics import YOLO
import cv2
import os

# Parameters
output_width = 640
output_height = 480
CONFIDENCE_THRESHOLD = 0.7
GREEN = (0, 255, 0)
RED = (0, 0, 255)
LINE_COLOR = (255, 255, 255)
BLUE = (255, 0, 0)

# Load the YOLO model
model_path = "./models/"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
model = YOLO(model_path).to('cpu')

# List of image paths
image_paths = [
    # Image for testing
]

def process_image(image_path):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    # Read and resize the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return
    image = cv2.resize(image, (output_width, output_height))

    # Run inference on the image
    results = model.predict(image, conf=CONFIDENCE_THRESHOLD, imgsz=(output_width, output_height))

    max_red_area = 0
    max_green_area = 0
    bgreen_tensor = None
    bred_tensor = None

    # Loop through the detected objects
    for box in results[0].boxes:
        w = box.xywh[0][2]
        h = box.xywh[0][3]
        area = max(w * w, h * h)

        # Determine if the object is "green" or "red"
        if int(box.cls[0]) == 1:  # Green class (adjust according to your model)
            if max_green_area < area:
                max_green_area = area
                bgreen_tensor = box
        elif int(box.cls[0]) == 0:  # Red class (adjust according to your model)
            if max_red_area < area:
                max_red_area = area
                bred_tensor = box

    # Draw rectangles and lines if objects are detected
    if bgreen_tensor is not None:
        x1_green, y1_green, x2_green, y2_green = map(int, bgreen_tensor.xyxy[0])
        cv2.rectangle(image, (x1_green, y1_green), (x2_green, y2_green), GREEN, 2)

    if bred_tensor is not None:
        x1_red, y1_red, x2_red, y2_red = map(int, bred_tensor.xyxy[0])
        cv2.rectangle(image, (x1_red, y1_red), (x2_red, y2_red), RED, 2)

    if bgreen_tensor and bred_tensor:
        bottom_x_y = (output_width // 2, output_height)
        line_x_top_center = (x2_green + x1_red) // 2
        line_y_top_center = (y1_green + y1_red) // 2
        line_center_x_y = (line_x_top_center, line_y_top_center)

        cv2.line(image, (x2_green, y1_green), (x1_red, y1_red), BLUE, 2)
        cv2.line(image, bottom_x_y, line_center_x_y, LINE_COLOR, 2)

    # Save and display the processed image
    output_path = image_path.replace(".png", "_detected.png")
    cv2.imwrite(output_path, image)
    cv2.imshow('Detections', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Process all images
for img_path in image_paths:
    process_image(img_path)
