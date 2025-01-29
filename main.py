from ultralytics import YOLO
import torch
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
model_path = "./models/yolo_model.pt"  # Pastikan model adalah file, bukan folder

if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(model_path).to(device)

# List of image paths
image_paths = [
    # Add image paths here
    # Example: "./images/sample.jpg"
    "./samples/"
]

def process_image(image_path, show_image=False):
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    # Read and resize the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to read image: {image_path}")
        return
    image = cv2.resize(image, (output_width, output_height))

    # Run inference on the image
    results = model.predict(image, conf=CONFIDENCE_THRESHOLD, imgsz=(output_width, output_height))

    max_red_area = 0
    max_green_area = 0
    bgreen_tensor = None
    bred_tensor = None

    # Loop through the detected objects
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                if box.xywh is not None and len(box.xywh) > 0:
                    x, y, w, h = map(float, box.xywh[0])
                    area = w * h

                    # Determine if the object is "green" or "red"
                    class_id = int(box.cls[0])
                    if class_id == 1:  # Green class
                        if area > max_green_area:
                            max_green_area = area
                            bgreen_tensor = box
                    elif class_id == 0:  # Red class
                        if area > max_red_area:
                            max_red_area = area
                            bred_tensor = box

    # Draw rectangles and lines if objects are detected
    if bgreen_tensor is not None and hasattr(bgreen_tensor, 'xyxy'):
        x1_green, y1_green, x2_green, y2_green = map(int, bgreen_tensor.xyxy[0])
        cv2.rectangle(image, (x1_green, y1_green), (x2_green, y2_green), GREEN, 2)
    else:
        x1_green = y1_green = x2_green = y2_green = None

    if bred_tensor is not None and hasattr(bred_tensor, 'xyxy'):
        x1_red, y1_red, x2_red, y2_red = map(int, bred_tensor.xyxy[0])
        cv2.rectangle(image, (x1_red, y1_red), (x2_red, y2_red), RED, 2)
    else:
        x1_red = y1_red = x2_red = y2_red = None

    # Draw line only if both objects are detected
    if None not in [x1_green, y1_green, x2_green, y2_green, x1_red, y1_red, x2_red, y2_red]:
        bottom_x_y = (output_width // 2, output_height)
        line_x_top_center = (x2_green + x1_red) // 2
        line_y_top_center = (y1_green + y1_red) // 2
        line_center_x_y = (line_x_top_center, line_y_top_center)

        # Draw lines
        cv2.line(image, (x2_green, y1_green), (x1_red, y1_red), BLUE, 2)
        cv2.line(image, bottom_x_y, line_center_x_y, LINE_COLOR, 2)

    # Save processed image
    output_path = image_path.replace(".png", "_detected.png").replace(".jpg", "_detected.jpg")
    cv2.imwrite(output_path, image)
    print(f"✅ Processed image saved: {output_path}")

    # Show image only if required
    if show_image:
        cv2.imshow('Detections', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    # Process all images
    for img_path in image_paths:
        process_image(img_path, show_image=False)  # Set to True to display images


if __name__ == "__main__":
    main()
