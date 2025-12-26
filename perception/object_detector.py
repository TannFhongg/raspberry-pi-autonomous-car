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
        Nhận diện vật thể - KHÔNG vẽ bounding box
        Chỉ trả về danh sách detections và frame gốc
        """
        if self.model is None or frame is None:
            return [], frame

        # Inference
        results = self.model(frame, imgsz=640, conf=self.conf_threshold, verbose=False)
        
        # Trích xuất thông tin detection (KHÔNG vẽ gì lên frame)
        detections = []

        for box in results[0].boxes:
            x, y, w, h = box.xywh[0].tolist()
            cls_id = int(box.cls[0])
            class_name = self.model.names[cls_id]
            conf = float(box.conf[0])
            
            detections.append({
                'class_name': class_name,
                'conf': conf,
                'x': x, 'y': y, 'w': w, 'h': h
            })

        return detections, frame