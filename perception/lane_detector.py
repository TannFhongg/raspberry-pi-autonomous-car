"""
Lane Detection Module (UPDATED for Picamera2)
Detects lane-like lines and calculates midline for following
Now supports BGR input from Picamera2
"""

import cv2
import numpy as np


def detect_line(frame, config=None):
    """
    Phát hiện đường line và tính toán lỗi lệch tâm.
    Phiên bản đã TUNE:
    - Hough Threshold: 20
    - Min Line Length: 30
    - Max Line Gap: 20 (Chặn nối điểm xa)
    - Slope Filter: > 0.5 (Chặn đường ngang)
    - Logic Offset: Xử lý khi chỉ thấy 1 vạch
    """
    # Cấu hình mặc định (đã được tune chuẩn theo kết quả test của bạn)
    if config is None:
        config = {
            'roi_top_ratio': 0.5,
            'roi_bottom_ratio': 1.0,
            'canny_low': 50,
            'canny_high': 150,
            'hough_threshold': 20,    # Đã chỉnh xuống 20
            'min_line_length': 30,    # Đã chỉnh xuống 30
            'max_line_gap': 20,       # Đã chỉnh xuống 20 (QUAN TRỌNG)
            'blur_kernel': 5,
        }

    height, width = frame.shape[:2]
    center_x = width // 2
    
    # Tạo ảnh debug
    frame_debug = frame.copy()
    cv2.line(frame_debug, (center_x, 0), (center_x, height), (0, 255, 255), 1) # Trục giữa xe (Vàng)

    # 1. Xử lý ảnh
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (config['blur_kernel'], config['blur_kernel']), 0)
    edges = cv2.Canny(blur, config['canny_low'], config['canny_high'])
    
    # 2. ROI (Vùng quan tâm)
    roi_top = int(height * config['roi_top_ratio'])
    roi_bottom = int(height * config['roi_bottom_ratio'])
    
    # Thu hẹp đỉnh hình thang một chút để tránh nhiễu 2 bên lề
    roi_vertices = np.array([[
        (0, roi_bottom),
        (int(width * 0.4), roi_top),  # Thu vào 0.4
        (int(width * 0.6), roi_top),  # Thu vào 0.6
        (width, roi_bottom)
    ]], dtype=np.int32)
    
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Vẽ ROI lên ảnh debug
    cv2.polylines(frame_debug, roi_vertices, True, (255, 0, 0), 2)

    # 3. Hough Transform
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=config['hough_threshold'],
        minLineLength=config['min_line_length'],
        maxLineGap=config['max_line_gap']
    )
    
    # Phân loại vạch trái/phải
    left_lines = []
    right_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            if x2 - x1 == 0:
                slope = 999.0 # Vô cực
            else:
                slope = (y2 - y1) / (x2 - x1)
            
            # --- BỘ LỌC ĐỘ DỐC (Đã sửa thành 0.5) ---
            if abs(slope) < 0.5:
                continue
            
            # Phân loại dựa trên vị trí và dấu của độ dốc
            # Vạch trái: nằm bên trái tâm VÀ slope âm (nghiêng /)
            # Vạch phải: nằm bên phải tâm VÀ slope dương (nghiêng \)
            if slope < 0 and x1 < center_x:
                left_lines.append((x1, y1, x2, y2))
            elif slope > 0 and x2 > center_x:
                right_lines.append((x1, y1, x2, y2))

    # 4. Tính toán vị trí vạch (Trung bình cộng)
    left_lane_x = None
    right_lane_x = None
    
    if left_lines:
        # Lấy trung bình tọa độ x tại đáy ảnh (y = height)
        # Công thức suy diễn: x = x1 + (height - y1) / slope
        x_bottoms = []
        for x1, y1, x2, y2 in left_lines:
            slope = (y2 - y1) / (x2 - x1)
            x_bottom = x1 + (height - y1) / slope
            x_bottoms.append(x_bottom)
            # Vẽ từng đoạn tìm được để debug
            cv2.line(frame_debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        left_lane_x = int(np.mean(x_bottoms))

    if right_lines:
        x_bottoms = []
        for x1, y1, x2, y2 in right_lines:
            slope = (y2 - y1) / (x2 - x1)
            x_bottom = x1 + (height - y1) / slope
            x_bottoms.append(x_bottom)
            cv2.line(frame_debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        right_lane_x = int(np.mean(x_bottoms))

    # 5. ===== LOGIC TÍNH TÂM ĐƯỜNG (SỬA ĐỔI QUAN TRỌNG NHẤT) =====
    
    # Giả định độ rộng đường (đo trên ảnh Straight debug của bạn)
    # Khoảng cách giữa 2 chấm xanh ~ 220 pixel
    LANE_WIDTH = 555 
    
    if left_lane_x is not None and right_lane_x is not None:
        # Trường hợp hoàn hảo: Thấy cả 2
        x_line = (left_lane_x + right_lane_x) // 2
        cv2.circle(frame_debug, (left_lane_x, height), 10, (0, 255, 0), -1)
        cv2.circle(frame_debug, (right_lane_x, height), 10, (0, 255, 0), -1)
        
    elif left_lane_x is not None:
        # Chỉ thấy TRÁI -> Tâm = Trái + (Rộng/2)
        x_line = left_lane_x + (LANE_WIDTH // 2)
        cv2.circle(frame_debug, (left_lane_x, height), 10, (0, 255, 0), -1)
        cv2.putText(frame_debug, "LEFT ONLY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
    elif right_lane_x is not None:
        # Chỉ thấy PHẢI -> Tâm = Phải - (Rộng/2)
        x_line = right_lane_x - (LANE_WIDTH // 2)
        cv2.circle(frame_debug, (right_lane_x, height), 10, (0, 255, 0), -1)
        cv2.putText(frame_debug, "RIGHT ONLY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
    else:
        # Mất cả 2 -> Giữ nguyên hướng cũ hoặc đi thẳng (Center)
        x_line = center_x
        cv2.putText(frame_debug, "NO LANE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Tính sai số
    error = x_line - center_x
    
    # Vẽ hướng đi
    cv2.line(frame_debug, (x_line, 0), (x_line, height), (255, 0, 255), 2) # Tâm đường ảo (Tím)
    cv2.arrowedLine(frame_debug, (center_x, height-20), (x_line, height-20), (0, 0, 255), 2)
    cv2.putText(frame_debug, f"Err: {error}", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return error, x_line, center_x, frame_debug


def detect_line_simple(frame):
    """
    Simplified lane detection using color thresholding
    Args:
        frame: BGR format from Picamera2
    """
    height, width = frame.shape[:2]
    center_x = width // 2
    
    frame_debug = frame.copy()
    
    # Convert to HSV (Input is BGR)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # <--- ĐÃ SỬA
    
    # Detect white/yellow lines
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    # ROI
    roi_top = height // 2
    mask_white[:roi_top, :] = 0
    
    contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            x_line = cx
            cv2.drawContours(frame_debug, [largest_contour], -1, (0, 255, 0), 2)
            cv2.circle(frame_debug, (cx, cy), 10, (255, 0, 255), -1)
        else:
            x_line = center_x
    else:
        x_line = center_x
    
    error = center_x - x_line
    cv2.line(frame_debug, (center_x, 0), (center_x, height), (0, 255, 255), 2)
    cv2.arrowedLine(frame_debug, (center_x, height - 50), (x_line, height - 50), (0, 0, 255), 3)
    cv2.putText(frame_debug, f"Error: {error:+d}", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return error, x_line, center_x, frame_debug