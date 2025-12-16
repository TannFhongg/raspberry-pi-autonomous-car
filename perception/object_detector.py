from ultralytics import YOLO
import logging
import os
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class ObjectDetector:
    def __init__(self, model_path='data/models/best_ncnn_model', conf_threshold=0.5):
        self.model = None
        self.conf_threshold = conf_threshold
        
        # Các ngưỡng bạn đang dùng trong robot_controller (để vẽ tham chiếu)
        self.ref_sizes = [120, 160, 200] 
        
        if os.path.exists(model_path):
            try:
                logger.info(f"Loading NCNN model from {model_path}...")
                self.model = YOLO(model_path, task='detect')
                logger.info("Model loaded successfully!")
                logger.info(f"Classes: {self.model.names}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
        else:
            logger.error(f"Model not found at {model_path}")

    def detect(self, frame):
        """
        Nhận diện vật thể và vẽ kích thước pixel
        """
        if self.model is None or frame is None:
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return [], frame

        # 1. Inference
        results = self.model(frame, imgsz=640, conf=self.conf_threshold, verbose=False)
        
        # 2. Lấy ảnh kết quả cơ bản từ YOLO (đã có khung bao và tên class)
        annotated_frame_rgb = results[0].plot()
        
        # 3. Chuyển sang BGR cho OpenCV/Web
        annotated_frame_bgr = cv2.cvtColor(annotated_frame_rgb, cv2.COLOR_RGB2BGR)
        
        # 4. Trích xuất thông tin và VẼ THÊM KÍCH THƯỚC
        detections = []
        height_img, width_img = annotated_frame_bgr.shape[:2]

        for box in results[0].boxes:
            # Lấy tọa độ tâm và kích thước
            x, y, w, h = box.xywh[0].tolist()
            x1, y1, x2, y2 = box.xyxy[0].tolist() # Tọa độ góc để vẽ chữ
            
            cls_id = int(box.cls[0])
            class_name = self.model.names[cls_id]
            conf = float(box.conf[0])
            
            # Kích thước lớn nhất (dùng để so sánh với ngưỡng)
            max_dim = max(w, h)
            
            detections.append({
                'class_name': class_name,
                'conf': conf,
                'x': x, 'y': y, 'w': w, 'h': h
            })

            # --- VẼ THÔNG SỐ PIXEL LÊN HÌNH ---
            # Nội dung: "Size: [Rộng]x[Cao] (Max: [Max])"
            # Ví dụ: "Size: 100x150 (Max: 150)"
            label = f"Size: {int(w)}x{int(h)} px"
            
            # Chọn màu chữ dựa trên kích thước (để biết đã đạt ngưỡng chưa)
            # Màu mặc định: Trắng
            text_color = (255, 255, 255) 
            if max_dim >= 160: # DIST_EXECUTE
                text_color = (0, 255, 0) # Xanh lá (Đã đạt ngưỡng thực thi)
            elif max_dim >= 120: # DIST_PREPARE
                text_color = (0, 255, 255) # Vàng (Đạt ngưỡng chuẩn bị)

            # Vẽ nền đen cho chữ dễ đọc
            cv2.rectangle(annotated_frame_bgr, 
                          (int(x1), int(y1) - 20), 
                          (int(x1) + 200, int(y1)), 
                          (0, 0, 0), -1)
            
            # Viết chữ lên hình
            cv2.putText(annotated_frame_bgr, label, 
                        (int(x1), int(y1) - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                        text_color, 2)

        # 5. (Tùy chọn) Vẽ thước kẻ tham chiếu bên cạnh phải màn hình
        # Để bạn dễ ước lượng kích thước mà không cần vật thể
        x_ref = width_img - 30
        y_center = height_img // 2
        
        # Vẽ vạch 120px (Vàng)
        cv2.line(annotated_frame_bgr, (x_ref, y_center - 60), (x_ref, y_center + 60), (0, 255, 255), 2)
        cv2.putText(annotated_frame_bgr, "120", (x_ref-35, y_center - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Vẽ vạch 160px (Xanh lá)
        cv2.line(annotated_frame_bgr, (x_ref + 10, y_center - 80), (x_ref + 10, y_center + 80), (0, 255, 0), 2)
        cv2.putText(annotated_frame_bgr, "160", (x_ref-25, y_center - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        return detections, annotated_frame_bgr