from ultralytics import YOLO
import cv2
import torch

# load model YOLO
models = "./models/bola aja.pt"

# Load Model YOLO dengan FP16 jika GPU tersedia
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(models).to(device)
model.fuse()

# Load video
video_path = "./samples/Sample video - Made with Clipchamp.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"❌ Gagal membuka video: {video_path}")
    exit()

# Ambil FPS dan resolusi video
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Resize frame untuk meningkatkan kecepatan pemrosesan dan pastikan kelipatan 32
new_width = (frame_width // 2) // 32 * 32
new_height = (frame_height // 2) // 32 * 32

# Inisialisasi Tracker (KCF untuk kecepatan)
tracker = cv2.TrackerKCF.create()
tracking = False
frame_counter = 0
inference_interval = 5  # Jalankan YOLO setiap 5 frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame untuk meningkatkan kecepatan pemrosesan
    frame_resized = cv2.resize(frame, (new_width, new_height))

    # Jalankan YOLO setiap beberapa frame
    if frame_counter % inference_interval == 0:
        results = model.predict(frame_resized, conf=0.7, half=True, imgsz=(new_width, new_height))
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1

                    # Pastikan bounding box valid
                    if w > 0 and h > 0 and x1 >= 0 and y1 >= 0 and x2 <= new_width and y2 <= new_height:
                        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        bbox = (x1, y1, w, h)

                        # Hanya mulai melacak jika belum melacak
                        if not tracking:
                            tracker.init(frame_resized, bbox)
                            tracking = True

    # Update tracker jika sedang melacak
    if tracking:
        success, bbox = tracker.update(frame_resized)
        if success:
            x, y, w, h = map(int, bbox)
            # Pastikan bounding box berada dalam frame
            if 0 <= x <= new_width and 0 <= y <= new_height and x + w <= new_width and y + h <= new_height:
                cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                tracking = False  # Reset tracking jika objek keluar dari frame
        else:
            tracking = False  # Reset tracking jika gagal update tracker

    frame_counter += 1
    cv2.imshow("Frame", frame_resized)

    # Tunggu untuk tombol 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan resource
cap.release()
cv2.destroyAllWindows()
