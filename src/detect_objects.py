import os
import cv2
from ultralytics import YOLO

# === SETTINGS ===
VIDEO_PATH = "C:/Users/DELL/Videos/football match.mp4"   
MODEL_PATH = "yolov8n.pt"                                # or trained weights
SAVE_FRAMES = False
FRAMES_DIR = "frames"
OUTPUT_VIDEO = "output_with_detections.mp4"

# === LOAD MODEL ===
model = YOLO(MODEL_PATH)

# === VIDEO CAPTURE ===
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open {VIDEO_PATH}")
    exit()

# === VIDEO WRITER ===
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

if SAVE_FRAMES:
    os.makedirs(FRAMES_DIR, exist_ok=True)

frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:   # end of video
        break

    # === YOLO DETECTION ===
    results = model(frame)              # normal call (not stream=True here)
    annotated_frame = results[0].plot() # draw boxes

    # write annotated frame to output video
    out.write(annotated_frame)

    # save raw frame
    if SAVE_FRAMES:
        frame_path = os.path.join(FRAMES_DIR, f"frame_{frame_id}.jpg")
        cv2.imwrite(frame_path, frame)

    # show on screen
    cv2.imshow("YOLO Football Detection", annotated_frame)

    # press 'q' to stop early
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_id += 1

cap.release()
out.release()
cv2.destroyAllWindows()
print("Output saved as:", OUTPUT_VIDEO)