import cv2
import time
import threading
from ultralytics import YOLO
from scene import build_scene_description, draw_description_bar


model = YOLO('yolov8n.pt') #loads model

#opens webcam, sets resolution to 640x480
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#Shared variables for frames and annotations
latest_frame = None
latest_annotated = None
latest_description = ""

#creates lock that prevents both threads from accessing shared variables
lock = threading.Lock()
prev_time = 0

#background thread function that runs detection on the latest frame and updates the annotated image
def detection_loop():
    global latest_annotated, latest_description
    while True:
        with lock:
            frame = latest_frame #safely reads latest frame using lock
        if frame is None:
            continue
        results = model(frame, imgsz=320, verbose=False) #runs detection on frame
        with lock:
            latest_annotated = results[0].plot() #draws detection boxes and updates latest annotated image using lock
            latest_description = build_scene_description(results, model) #builds scene description from results

thread = threading.Thread(target=detection_loop, daemon=True)#daemon means that it dies when program dies
thread.start()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    with lock:
        latest_frame = frame
        annotated = latest_annotated #updates the shared frame for the detection thread to pick up and grab the latest annotated frame to display
        description = latest_description #grabs detection

    if annotated is not None:
        curr_time = time.time()
        fps = 1/(curr_time - prev_time) #calculates FPS
        prev_time = curr_time
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        annotated = draw_description_bar(annotated, description)

        cv2.imshow('YOLOv8 Detection', annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()