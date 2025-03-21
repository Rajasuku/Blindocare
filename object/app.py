from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import pyttsx3
from queue import Queue
from threading import Thread
from ultralytics import YOLO
import time

app = Flask(__name__)

# Load YOLO Model
model_path = "gpModel.pt"
try:
    model = YOLO(model_path)
    print("YOLO model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

# Queue for Text-to-Speech Processing
queue = Queue()

def speak(q):
    """Thread function for text-to-speech."""
    engine = pyttsx3.init()
    engine.setProperty('rate', 200)
    engine.setProperty('volume', 1.0)

    while True:
        if not q.empty():
            objects = q.get()
            for label, distance, position in objects:
                rounded_distance = round(distance * 2) / 2  
                rounded_distance_str = str(int(rounded_distance)) if rounded_distance.is_integer() else str(rounded_distance)
                engine.say(f"{label} is {rounded_distance_str} meters on {position}")
                engine.runAndWait()
            with q.mutex:
                q.queue.clear()
        else:
            time.sleep(0.1)

# Start TTS Thread
t = Thread(target=speak, args=(queue,))
t.daemon = True
t.start()

# Initialize Camera
def initialize_camera():
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera accessed on index {i}")
            return cap
    print("Error: Unable to access any camera.")
    exit()

cap = initialize_camera()

def get_position(frame_width, box):
    """Determine object's position (LEFT, FORWARD, RIGHT)."""
    if box[0] < frame_width // 3:
        return "LEFT"
    elif box[0] < 2 * (frame_width // 3):
        return "FORWARD"
    else:
        return "RIGHT"

def generate_frames():
    """Generate frames for frontend streaming."""
    while True:
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)  

        # Run YOLO object detection
        results = model.predict(frame, imgsz=640, conf=0.3)

        detected_objects = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                object_width = x2 - x1
                distance = 500 / (object_width + 1e-6)
                position = get_position(frame.shape[1], [x1, y1, x2, y2])

                detected_objects.append((label, round(distance, 1), position))

                # Draw Bounding Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{label} - {round(distance, 1)}m", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if detected_objects:
            queue.put(detected_objects)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Route to stream video feed."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_objects')
def get_objects():
    """Returns detected objects."""
    return jsonify(list(queue.queue))

if __name__ == '__main__':
    app.run(debug=True)
