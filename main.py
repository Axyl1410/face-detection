import cv2
import os
from ultralytics import YOLO

# Khai báo mô hình
MODEL_TRACKER = "yolo11n-seg.pt"
MODEL_FACE = "yolov11m-face.pt"
VIDEO_SOURCE = "video2.mp4"  # Sử dụng camera

# Kiểm tra mô hình tồn tại không
if not os.path.exists(MODEL_TRACKER) or not os.path.exists(MODEL_FACE):
    raise FileNotFoundError("Không tìm thấy tệp mô hình YOLO!")

# Load các mô hình YOLO
tracker_model = YOLO(MODEL_TRACKER)
face_model = YOLO(MODEL_FACE)

# Mở camera
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    raise ValueError("Không thể mở camera!")

# Thư mục lưu ảnh khuôn mặt
SAVE_DIR = "faces"
os.makedirs(SAVE_DIR, exist_ok=True)

# Biến trạng thái
tracking_id = None  # ID cần theo dõi
frame_count = 0  # Đếm frame để đặt tên ảnh
last_saved_time = 0  # Kiểm soát lưu ảnh khuôn mặt

print("Camera đang chạy... Nhấn 'V' để nhập ID cần theo dõi.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Nhận diện tất cả đối tượng
    results = tracker_model.track(frame, persist=True)

    if tracking_id is None:
        # Nếu chưa chọn ID, chỉ hiển thị người
        for r in results:
            for box in r.boxes:
                if box.cls == 0:  # Chỉ lấy class "person"
                    track_id = box.id
                    if track_id is not None:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"ID: {int(track_id)}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        # Nếu đã chọn ID, chỉ theo dõi người đó
        for r in results:
            for box in r.boxes:
                if box.cls == 0:  # Chỉ lấy class "person"
                    track_id = box.id
                    if track_id is not None and int(track_id) == tracking_id:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        # Cắt ảnh người theo dõi
                        person_crop = frame[y1:y2, x1:x2]

                        # Nhận diện khuôn mặt
                        face_results = face_model(person_crop)
                        for face in face_results:
                            for fbox in face.boxes:
                                fx1, fy1, fx2, fy2 = map(int, fbox.xyxy[0])
                                face_crop = person_crop[fy1:fy2, fx1:fx2]

                                # Vẽ bounding box khuôn mặt
                                cv2.rectangle(frame, (x1 + fx1, y1 + fy1), (x1 + fx2, y1 + fy2), (0, 0, 255), 2)
                                cv2.putText(frame, "Face", (x1 + fx1, y1 + fy1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                                # Lưu ảnh khuôn mặt khi nhấn 'C'
                                if cv2.waitKey(1) & 0xFF == ord('c'):
                                    if frame_count - last_saved_time > 10:  # Kiểm soát lưu (mỗi 10 frame)
                                        filename = os.path.join(SAVE_DIR, f"face_{frame_count}.jpg")
                                        cv2.imwrite(filename, face_crop)
                                        print(f"Ảnh khuôn mặt đã lưu: {filename}")
                                        last_saved_time = frame_count

                        # Vẽ bounding box cho người theo dõi
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, f"Tracking ID: {tracking_id}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Hiển thị video
    cv2.imshow("Real-Time Tracking", frame)

    # Nhấn 'V' để nhập ID cần theo dõi
    key = cv2.waitKey(1) & 0xFF
    if key == ord('v'):
        try:
            tracking_id = int(input("Nhập ID cần theo dõi: "))
            print(f"Đang theo dõi ID: {tracking_id}")
        except ValueError:
            print("ID không hợp lệ! Vui lòng nhập số.")

    # Nhấn 'Q' để thoát
    if key == ord('q'):
        break

    frame_count += 1  # Đếm số frame

# Dọn dẹp tài nguyên
cap.release()
cv2.destroyAllWindows()
