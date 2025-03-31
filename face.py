import cv2
import pyodbc
from deepface import DeepFace

# Kết nối SQL Server (Cập nhật thông tin theo hệ thống của bạn)
conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=Admin\SQLEXPRESS;"
    "DATABASE=face;"
    "UID=minhduy;"
    "PWD=123456;"
)
cursor = conn.cursor()

# Thư mục chứa ảnh trong database
db_path = "Database"

# Mở webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Lưu ảnh tạm thời để phân tích
    temp_image = "temp.jpg"
    cv2.imwrite(temp_image, frame)

    try:
        # Nhận diện khuôn mặt bằng Facenet512
        result = DeepFace.find(img_path=temp_image, db_path=db_path, model_name="Facenet512", enforce_detection=False)

        if len(result) > 0 and not result[0].empty:
            matched_face = result[0].iloc[0]["identity"].split("/")[-1]  # Lấy tên file ảnh

            # Truy vấn thông tin từ SQL Server
            query = f"SELECT name, age, gender, position FROM employees WHERE image_path = ?"
            cursor.execute(query, (matched_face,))
            user_info = cursor.fetchone()

            if user_info:
                display_text = f"Name: {user_info[0]}, Age: {user_info[1]}, Gender: {user_info[2]}, Position: {user_info[3]}"
            else:
                display_text = "Không tìm thấy thông tin!"
        else:
            display_text = "Không nhận diện được khuôn mặt!"

        # Hiển thị thông tin lên màn hình
        cv2.putText(frame, display_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    except Exception as e:
        print("Lỗi:", e)

    # Hiển thị hình ảnh từ webcam
    cv2.imshow("Face Recognition - Facenet512 + SQL", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Đóng kết nối và giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
cursor.close()
conn.close()
