"""
Lane Detection Module - FIXED for 1640x1232 → 640x480 resize
OPTIMIZED for BLACK LINES on WHITE BACKGROUND
Designed for: 38cm lane width, 15cm robot width, black tape lines
Camera: Raspberry Pi Camera Module 2
"""

import cv2
import numpy as np


def detect_line(frame, config=None):
    """
    Phát hiện vạch KẺ ĐEN trên nền TRẮNG (bìa trắng)
    
    Thông số thực tế:
    - Lane width: 38cm
    - Robot width: 15cm
    - Input: Any resolution (sẽ tự động resize về 640x480)
    - Line color: BLACK on WHITE background
    """
    
    # ============================================================
    # BƯỚC 0: RESIZE VỀ 640x480 CHUẨN (Quan trọng!)
    # ============================================================
    original_height, original_width = frame.shape[:2]
    
    # Nếu không phải 640x480, resize về chuẩn
    if original_width != 640 or original_height != 480:
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
        print(f"[INFO] Resized from {original_width}x{original_height} to 640x480")
    
    # ============================================================
    # Cấu hình mặc định - TUNED cho vạch đen trên nền trắng
    # ============================================================
    if config is None:
        config = {
            'roi_top_ratio': 0.5,      # BẮT ĐẦU THẤP HƠN (35% thay vì 40%) - Nhìn GẦN XE HƠN
            'roi_bottom_ratio': 1.0,
            'canny_low': 80,             # TĂNG lên 40 (nền trắng sạch, cần ngưỡng cao hơn)
            'canny_high': 185,           # TĂNG lên 120
            'hough_threshold': 45,       # TĂNG lên 20 (vạch rõ hơn trên nền trắng)
            'min_line_length': 60,       # TĂNG lên 30 (loại nhiễu)
            'max_line_gap': 30,          # TĂNG lên 20
            'blur_kernel': 7,            # GIẢM về 5 (nền trắng ít nhiễu hơn nền nhà)
        }

    height, width = frame.shape[:2]  # Giờ luôn là 640x480
    center_x = width // 2
    
    # ============================================================
    # LANE WIDTH PIXELS - ĐÃ CALIBRATE CHO 640x480
    # ============================================================
    # Công thức ước tính:
    # - Camera nhìn từ trên cao ~20cm, góc nhìn ~62 degrees (Camera V2)
    # - Tại đáy ảnh (gần xe), 38cm lane ≈ 200-250 pixels
    # QUAN TRỌNG: Cần chạy calibration để lấy số chính xác!
    
    LANE_WIDTH_PIXELS = 245  # ⚠️ GIÁ TRỊ ƯỚC TÍNH - PHẢI CALIBRATE!
    
    # Debug frame
    frame_debug = frame.copy()
    cv2.line(frame_debug, (center_x, 0), (center_x, height), (0, 255, 255), 2)

    # ============================================================
    # 1. TIỀN XỬ LÝ ẢNH - CHO NỀN TRẮNG, VẠCH ĐEN
    # ============================================================
    
    # Chuyển sang grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # ĐẢO NGƯỢC: Vạch đen → trắng (Canny hoạt động tốt hơn)
    gray_inverted = cv2.bitwise_not(gray)
    
    # Làm mờ nhẹ (nền trắng ít nhiễu hơn nền nhà)
    blur = cv2.GaussianBlur(gray_inverted, (config['blur_kernel'], config['blur_kernel']), 0)
    
    # TĂNG CƯỜNG TƯƠNG PHẢN (Optional - có thể bỏ nếu nền trắng đồng đều)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))  # Giảm clipLimit xuống 1.5
    enhanced = clahe.apply(blur)
    
    # Canny edge detection
    edges = cv2.Canny(enhanced, config['canny_low'], config['canny_high'])
    
    # ============================================================
    # 2. XÁC ĐỊNH ROI - HÌNH THANG RỘNG HƠN
    # ============================================================
    roi_top = int(height * config['roi_top_ratio'])
    roi_bottom = int(height * config['roi_bottom_ratio'])
    
    # Mở rộng ROI (30%-70% thay vì 35%-65%) - Bắt vạch ở 2 bên tốt hơn
    roi_vertices = np.array([[
        (0, roi_bottom),
        (int(width * 0.2), roi_top),  # MỞ RỘNG: 30% thay vì 35%
        (int(width * 0.8), roi_top),  # MỞ RỘNG: 70% thay vì 65%
        (width, roi_bottom)
    ]], dtype=np.int32)
    
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Vẽ ROI lên debug frame
    cv2.polylines(frame_debug, roi_vertices, True, (255, 0, 0), 2)

    # ============================================================
    # 3. HOUGH TRANSFORM
    # ============================================================
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=config['hough_threshold'],
        minLineLength=config['min_line_length'],
        maxLineGap=config['max_line_gap']
    )
    
    # ============================================================
    # 4. PHÂN LOẠI VẠCH TRÁI/PHẢI - LOGIC CẢI THIỆN
    # ============================================================
    left_lines = []
    right_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Tính độ dốc
            if abs(x2 - x1) < 1:  # Tránh chia cho 0
                continue
            
            slope = (y2 - y1) / (x2 - x1)
            
            # BỘ LỌC ĐỘ DỐC: Chặt chẽ hơn để loại nhiễu
            if abs(slope) < 0.5:  # TĂNG từ 0.4 lên 0.5
                continue
            
            # Tính điểm giữa để phân loại
            mid_x = (x1 + x2) / 2
            
            # Phân loại: Vạch TRÁI (slope âm, nằm bên trái tâm)
            if slope < -0.5 and mid_x < center_x:
                left_lines.append((x1, y1, x2, y2, slope))
            # Phân loại: Vạch PHẢI (slope dương, nằm bên phải tâm)
            elif slope > 0.5 and mid_x > center_x:
                right_lines.append((x1, y1, x2, y2, slope))
    
    # ============================================================
    # 5. TÍNH TOÁN VỊ TRÍ VẠCH (Extrapolate về đáy ảnh)
    # ============================================================
    left_lane_x = None
    right_lane_x = None
    
    def calculate_lane_x(lines, color):
        """Tính tọa độ x tại đáy ảnh từ danh sách lines"""
        if not lines:
            return None
        
        x_bottoms = []
        slopes_valid = []
        
        for x1, y1, x2, y2, slope in lines:
            # Ngoại suy đến đáy ảnh: x_bottom = x1 + (height - y1) / slope
            x_bottom = x1 + (height - y1) / slope
            
            # Kiểm tra x_bottom có hợp lý không (trong khoảng 0 - width)
            if 0 <= x_bottom <= width:
                x_bottoms.append(x_bottom)
                slopes_valid.append(slope)
                # Vẽ line để debug
                cv2.line(frame_debug, (x1, y1), (x2, y2), color, 2)
        
        if x_bottoms:
            # Lấy MEDIAN thay vì MEAN (chống outlier tốt hơn)
            return int(np.median(x_bottoms))
        return None
    
    left_lane_x = calculate_lane_x(left_lines, (0, 255, 0))    # Xanh lá
    right_lane_x = calculate_lane_x(right_lines, (255, 0, 0))  # Xanh dương

    # ============================================================
    # 6. LOGIC TÍNH TÂM ĐƯỜNG (3 trường hợp)
    # ============================================================
    
    lane_status = "UNKNOWN"
    
    if left_lane_x is not None and right_lane_x is not None:
        # CASE 1: Thấy cả 2 vạch - HOÀN HẢO
        x_line = (left_lane_x + right_lane_x) // 2
        lane_status = "BOTH_LANES"
        
        # Vẽ 2 điểm vạch
        cv2.circle(frame_debug, (left_lane_x, height - 10), 10, (0, 255, 0), -1)
        cv2.circle(frame_debug, (right_lane_x, height - 10), 10, (255, 0, 0), -1)
        cv2.putText(frame_debug, "BOTH LANES", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
    elif left_lane_x is not None:
        # CASE 2: Chỉ thấy TRÁI
        x_line = left_lane_x + (LANE_WIDTH_PIXELS // 2)
        lane_status = "LEFT_ONLY"
        
        cv2.circle(frame_debug, (left_lane_x, height - 10), 10, (0, 255, 0), -1)
        cv2.putText(frame_debug, "LEFT ONLY", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
    elif right_lane_x is not None:
        # CASE 3: Chỉ thấy PHẢI
        x_line = right_lane_x - (LANE_WIDTH_PIXELS // 2)
        lane_status = "RIGHT_ONLY"
        
        cv2.circle(frame_debug, (right_lane_x, height - 10), 10, (255, 0, 0), -1)
        cv2.putText(frame_debug, "RIGHT ONLY", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
    else:
        # CASE 4: Mất cả 2
        x_line = center_x
        lane_status = "NO_LANE"
        forced_error = 999
        cv2.putText(frame_debug, "NO LANE DETECTED", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # ============================================================
    # 7. TÍNH SAI SỐ VÀ VẼ DEBUG
    # ============================================================
    CAMERA_OFFSET = -0  # Hiệu chỉnh nếu camera không đặt chính giữa robot
    if lane_status == "NO_LANE":
        error = 999  # ⚠️ QUAN TRỌNG: Gán cứng lỗi 999 khi mất line
    else:
        error = x_line - center_x + CAMERA_OFFSET  # Các trường hợp còn lại tính toán bình thường
    
    # Vẽ đường tâm dự đoán (màu tím)
    cv2.line(frame_debug, (x_line, 0), (x_line, height), (255, 0, 255), 3)
    
    # Vẽ mũi tên chỉ hướng điều chỉnh
    arrow_y = height - 50
    cv2.arrowedLine(frame_debug, (center_x, arrow_y), 
                    (x_line, arrow_y), (0, 0, 255), 4, tipLength=0.3)
    
    # Hiển thị thông tin chi tiết
    info_y = height - 10
    cv2.putText(frame_debug, f"Error: {error:+4d}px | Status: {lane_status}", 
                (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Thêm thông tin debug bổ sung (góc trên phải)
    info_lines = [
        f"Left: {left_lane_x if left_lane_x else 'None'}",
        f"Right: {right_lane_x if right_lane_x else 'None'}",
        f"Lane Width: {LANE_WIDTH_PIXELS}px"
    ]
    
    for i, line_text in enumerate(info_lines):
        cv2.putText(frame_debug, line_text, 
                    (width - 220, 30 + i*25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    return error, x_line, center_x, frame_debug


def detect_line_black_adaptive(frame):
    """
    Phương pháp dự phòng: ADAPTIVE THRESHOLD
    Tốt hơn khi ánh sáng không đều hoặc Hough thất bại
    """
    # Resize về 640x480 nếu cần
    if frame.shape[1] != 640:
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
    
    height, width = frame.shape[:2]
    center_x = width // 2
    LANE_WIDTH_PIXELS = 240  # Cùng giá trị với detect_line()
    
    frame_debug = frame.copy()
    
    # Chuyển sang grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Làm mờ
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # ADAPTIVE THRESHOLD - Vạch đen thành trắng
    thresh = cv2.adaptiveThreshold(
        blur, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        blockSize=21,  # TĂNG lên 21 (phù hợp với nền trắng lớn)
        C=8            # TĂNG lên 8
    )
    
    # ROI - Chỉ xét 2/3 dưới ảnh
    roi_top = int(height * 0.35)
    thresh[:roi_top, :] = 0
    
    # Tìm contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Lọc contours theo diện tích
    valid_contours = [c for c in contours if cv2.contourArea(c) > 200]  # TĂNG lên 200
    
    if len(valid_contours) >= 2:
        # Sắp xếp theo vị trí x
        valid_contours = sorted(valid_contours, key=lambda c: cv2.boundingRect(c)[0])
        
        # Lấy 2 contours ngoài cùng
        left_contour = valid_contours[0]
        right_contour = valid_contours[-1]
        
        # Tính tâm
        M_left = cv2.moments(left_contour)
        M_right = cv2.moments(right_contour)
        
        if M_left['m00'] > 0 and M_right['m00'] > 0:
            left_x = int(M_left['m10'] / M_left['m00'])
            right_x = int(M_right['m10'] / M_right['m00'])
            
            x_line = (left_x + right_x) // 2
            
            cv2.drawContours(frame_debug, [left_contour], -1, (0, 255, 0), 2)
            cv2.drawContours(frame_debug, [right_contour], -1, (255, 0, 0), 2)
            cv2.putText(frame_debug, "BOTH LANES (Adaptive)", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            x_line = center_x
            
    elif valid_contours:
        # Chỉ thấy 1 vạch
        M = cv2.moments(valid_contours[0])
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            
            # Dự đoán: Nếu contour ở bên trái → thêm nửa lane width
            if cx < center_x:
                x_line = cx + (LANE_WIDTH_PIXELS // 2)
                cv2.putText(frame_debug, "LEFT ONLY (Adaptive)", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                x_line = cx - (LANE_WIDTH_PIXELS // 2)
                cv2.putText(frame_debug, "RIGHT ONLY (Adaptive)", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.drawContours(frame_debug, [valid_contours[0]], -1, (0, 255, 0), 2)
        else:
            x_line = center_x
    else:
        x_line = center_x
        cv2.putText(frame_debug, "NO LANE (Adaptive)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    error = x_line - center_x
    
    # Vẽ debug
    cv2.line(frame_debug, (center_x, 0), (center_x, height), (0, 255, 255), 2)
    cv2.line(frame_debug, (x_line, 0), (x_line, height), (255, 0, 255), 3)
    cv2.arrowedLine(frame_debug, (center_x, height - 50), 
                    (x_line, height - 50), (0, 0, 255), 4)
    cv2.putText(frame_debug, f"Error: {error:+d}px", 
                (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return error, x_line, center_x, frame_debug


def calibrate_lane_width(frame, show_result=False):
    """
    Calibration tool - Đo 38cm lane thành pixels
    Đã sửa: KHÔNG dùng cv2.imshow() (không có màn hình)
    """
    # Resize về 640x480 nếu cần
    if frame.shape[1] != 640:
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
    
    height, width = frame.shape[:2]
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_inv = cv2.bitwise_not(gray)
    edges = cv2.Canny(gray_inv, 40, 120)
    
    # Chỉ xét 20% đáy ảnh
    edges[:int(height * 0.8), :] = 0
    
    # Tìm đường thẳng
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 25, minLineLength=35, maxLineGap=20)
    
    frame_calib = frame.copy()
    
    if lines is not None:
        # Vẽ tất cả lines
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame_calib, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Tìm x min và max
        x_coords = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            x_coords.extend([x1, x2])
        
        if x_coords:
            left_x = min(x_coords)
            right_x = max(x_coords)
            lane_width_pixels = right_x - left_x
            
            cv2.circle(frame_calib, (left_x, height - 10), 10, (0, 255, 0), -1)
            cv2.circle(frame_calib, (right_x, height - 10), 10, (255, 0, 0), -1)
            cv2.line(frame_calib, (left_x, height - 30), 
                     (right_x, height - 30), (255, 255, 0), 3)
            
            cv2.putText(frame_calib, f"Lane: {lane_width_pixels}px = 38cm", 
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # LƯU ẢNH thay vì hiển thị
            cv2.imwrite("calibration_result.jpg", frame_calib)
            
            print(f"\n{'='*60}")
            print(f"✅ CALIBRATION THÀNH CÔNG:")
            print(f"  Lane Width (Real):  38 cm")
            print(f"  Lane Width (Pixel): {lane_width_pixels} px")
            print(f"  Scale Factor:       {38 / lane_width_pixels:.4f} cm/px")
            print(f"  📸 Đã lưu: calibration_result.jpg")
            print(f"{'='*60}\n")
            print(f"⚠️  CẬP NHẬT NGAY:")
            print(f"  Sửa dòng 53 trong lane_detector.py:")
            print(f"  LANE_WIDTH_PIXELS = {lane_width_pixels}")
            print(f"{'='*60}\n")
            
            return lane_width_pixels
    
    print("❌ Không tìm thấy 2 vạch để calibrate!")
    print("💡 Kiểm tra:")
    print("  - Xe có đang ở giữa lane không?")
    print("  - Vạch đen có rõ ràng trên nền trắng không?")
    print("  - Ánh sáng có đủ không?")
    return None