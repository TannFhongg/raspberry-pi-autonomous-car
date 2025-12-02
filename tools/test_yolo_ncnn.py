from ultralytics import YOLO
import cv2

# Load mô hình NCNN (trỏ vào THƯ MỤC chứa file ncnn, không phải file lẻ)
model = YOLO('data/models/best_ncnn_model', task='detect')

# Khởi tạo camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Nhận diện (Inference)
    results = model(frame, imgsz=640)

    # Vẽ kết quả lên khung hình
    annotated_frame = results[0].plot()

    # Hiển thị
    cv2.imshow("YOLO11 NCNN Inference", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()