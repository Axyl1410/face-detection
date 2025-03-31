import cv2
import os
import time
import threading
from flask import Flask, Response, request, jsonify
from ultralytics import YOLO

# Khởi tạo Flask
app = Flask(__name__)

# Load mô hình YOLO
MODEL_TRACKER = "yolo11n-seg.pt"
MODEL_FACE = "yolov11m-face.pt"
tracker_model = YOLO(MODEL_TRACKER)
face_model = YOLO(MODEL_FACE)

# Khởi tạo camera
cap = cv2.VideoCapture(1)
tracking_id = None  # ID cần theo dõi
frame_lock = threading.Lock()
save_dir = "faces"
os.makedirs(save_dir, exist_ok=True)


def generate_frames():
    global tracking_id
    while True:
        with frame_lock:
            success, frame = cap.read()
            if not success:
                break

            results = tracker_model.track(frame, persist=True)

            for r in results:
                for box in r.boxes:
                    if box.cls == 0:  # Chỉ lấy class "person"
                        track_id = box.id
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        color = (0, 255, 0) if tracking_id is None else (255, 0, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"ID: {int(track_id)}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/track', methods=['POST'])
def set_tracking_id():
    global tracking_id
    data = request.json
    tracking_id = data.get('id')
    return jsonify({"message": f"Tracking ID set to {tracking_id}"})


@app.route('/save_face', methods=['POST'])
def save_face():
    global tracking_id
    success, frame = cap.read()
    if not success or tracking_id is None:
        return jsonify({"error": "No frame captured or no tracking ID"})

    face_results = face_model(frame)
    for face in face_results:
        for fbox in face.boxes:
            x1, y1, x2, y2 = map(int, fbox.xyxy[0])
            face_crop = frame[y1:y2, x1:x2]
            filename = os.path.join(save_dir, f"face_{int(time.time())}.jpg")
            cv2.imwrite(filename, face_crop)
            return jsonify({"message": f"Face saved at {filename}"})

    return jsonify({"error": "No face detected"})


@app.route('/stop', methods=['POST'])
def stop_camera():
    cap.release()
    return jsonify({"message": "Camera stopped"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
