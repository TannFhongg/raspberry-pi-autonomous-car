import cv2
import numpy as np

class VisualOdometry:
    def __init__(self):
        self.prev_gray = None
        self.features = None
        
        # Thông số Lucas-Kanade Optical Flow
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        # Thông số Good Features to Track
        self.feature_params = dict(maxCorners=100,
                                   qualityLevel=0.3,
                                   minDistance=7,
                                   blockSize=7)
        
        self.total_distance_y = 0.0 # Quãng đường tiến/lùi (pixel)

    def process_frame(self, frame):
        """
        Tính toán độ dịch chuyển giữa frame hiện tại và frame trước
        Trả về: dy (độ dịch chuyển theo trục dọc - tiến/lùi)
        """
        # Chuyển sang xám
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        dy = 0.0
        
        if self.prev_gray is not None:
            # Nếu chưa có điểm đặc trưng, tìm điểm mới
            if self.features is None or len(self.features) < 10:
                self.features = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)
            
            if self.features is not None:
                # Tính Optical Flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.features, None, **self.lk_params)
                
                # Chọn các điểm tốt (status = 1)
                if p1 is not None:
                    good_new = p1[st == 1]
                    good_old = self.features[st == 1]
                    
                    # Tính trung bình độ dịch chuyển Y (Trục dọc)
                    # y cũ - y mới (Nếu xe tiến, ảnh trôi xuống -> y mới > y cũ -> dy âm)
                    # Ta muốn: Xe tiến -> Distance tăng
                    movements = good_old[:, 1] - good_new[:, 1] 
                    
                    if len(movements) > 0:
                        dy = np.mean(movements)
                        
                    # Cập nhật điểm để track tiếp
                    self.features = good_new.reshape(-1, 1, 2)
        
        self.prev_gray = gray.copy()
        self.total_distance_y += dy
        
        return dy
        
    def get_total_distance(self):
        return self.total_distance_y
    
    def reset(self):
        self.total_distance_y = 0.0
        self.prev_gray = None
        self.features = None