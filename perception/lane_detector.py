"""
Lane Detection Module - OPTIMIZED for BLACK LINES on LIGHT BACKGROUND
Designed for: 38cm lane width, 15cm robot width, black tape lines
Camera: Raspberry Pi Camera Module 2 at 640x480
"""

import cv2
import numpy as np


def detect_line(frame, config=None):
    """
    Phát hiện vạch KẺ ĐEN trên nền SÁNG (nhà/trắng)
    
    Thông số thực tế:
    - Lane width: 38cm
    - Robot width: 15cm
    - Resolution: 640x480
    - Line color: BLACK
    """
    if frame.shape[1] != 640:
        frame = cv2.resize(frame, (640, 480))
    # Cấu hình mặc định được tối ưu cho vạch đen
    if config is None:
        config = {
            'roi_top_ratio': 0.4,      # Bắt đầu từ 40% chiều cao (gần xe hơn)
            'roi_bottom_ratio': 1.0,
            'canny_low': 30,            # Giảm để bắt được cạnh mờ
            'canny_high': 100,
            'hough_threshold': 15,      # Giảm để nhạy hơn
            'min_line_length': 25,      # Giảm để bắt vạch ngắn
            'max_line_gap': 15,
            'blur_kernel': 7,           # Tăng để giảm nhiễu nền nhà
        }

    height, width = frame.shape[:2]
    center_x = width // 2
    
    # === BƯỚC QUAN TRỌNG: TÍNH TOÁN LANE WIDTH THỰC TẾ ===
    # Giả sử camera nhìn thẳng, tại đáy ảnh (gần xe):
    # - Nếu lane 38cm chiếm X pixels ở giữa frame
    # - Cần calibrate bằng cách đo thực tế hoặc dùng test ảnh
    # Ví dụ: Nếu 38cm = 250 pixels (cần đo lại!)
    LANE_WIDTH_PIXELS = 280  # ⚠️ CẦN CALIBRATE BẰNG ẢNH THẬT
    
    # Debug frame
    frame_debug = frame.copy()
    cv2.line(frame_debug, (center_x, 0), (center_x, height), (0, 255, 255), 1)

    # ============================================================
    # 1. TIỀN XỬ LÝ ẢNH - TỐI ƯU CHO VẠCH ĐEN
    # ============================================================
    
    # Chuyển sang grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # QUAN TRỌNG: ĐẢO NGƯỢC ẢNH (Vạch đen thành trắng để Canny detect tốt hơn)
    gray_inverted = cv2.bitwise_not(gray)
    
    # Làm mờ để giảm nhiễu
    blur = cv2.GaussianBlur(gray_inverted, (config['blur_kernel'], config['blur_kernel']), 0)
    
    # TĂNG CƯỜNG TƯƠNG PHẢN (làm vạch nổi bật hơn)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blur)
    
    # Canny edge detection
    edges = cv2.Canny(enhanced, config['canny_low'], config['canny_high'])
    
    # ============================================================
    # 2. XÁC ĐỊNH ROI (Region of Interest)
    # ============================================================
    roi_top = int(height * config['roi_top_ratio'])
    roi_bottom = int(height * config['roi_bottom_ratio'])
    
    # Hình thang ROI - thu hẹp phía trên (xa xe)
    roi_vertices = np.array([[
        (0, roi_bottom),
        (int(width * 0.35), roi_top),  # Thu vào 35%
        (int(width * 0.65), roi_top),  # Thu vào 65%
        (width, roi_bottom)
    ]], dtype=np.int32)
    
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Vẽ ROI lên debug frame
    cv2.polylines(frame_debug, roi_vertices, True, (255, 0, 0), 2)

    # ============================================================
    # 3. HOUGH TRANSFORM - PHÁT HIỆN ĐƯỜNG THẲNG
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
    # 4. PHÂN LOẠI VẠCH TRÁI/PHẢI
    # ============================================================
    left_lines = []
    right_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Tính độ dốc
            if x2 - x1 == 0:
                slope = 999.0
            else:
                slope = (y2 - y1) / (x2 - x1)
            
            # BỘ LỌC: Chỉ giữ đường có độ dốc > 0.4 (loại bỏ đường ngang)
            if abs(slope) < 0.4:
                continue
            
            # Phân loại theo vị trí và độ dốc
            mid_x = (x1 + x2) / 2
            
            if slope < 0 and mid_x < center_x:  # Vạch trái
                left_lines.append((x1, y1, x2, y2, slope))
            elif slope > 0 and mid_x > center_x:  # Vạch phải
                right_lines.append((x1, y1, x2, y2, slope))
    
    # ============================================================
    # 5. TÍNH TOÁN VỊ TRÍ VẠCH (tại đáy ảnh)
    # ============================================================
    left_lane_x = None
    right_lane_x = None
    
    def calculate_lane_x(lines, color):
        """Tính tọa độ x tại đáy ảnh từ danh sách lines"""
        if not lines:
            return None
        
        x_bottoms = []
        for x1, y1, x2, y2, slope in lines:
            # Ngoại suy đến đáy ảnh: x = x1 + (height - y1) / slope
            x_bottom = x1 + (height - y1) / slope
            
            # Kiểm tra x_bottom có hợp lý không (trong khoảng 0 - width)
            if 0 <= x_bottom <= width:
                x_bottoms.append(x_bottom)
                # Vẽ line để debug
                cv2.line(frame_debug, (x1, y1), (x2, y2), color, 2)
        
        if x_bottoms:
            return int(np.mean(x_bottoms))
        return None
    
    left_lane_x = calculate_lane_x(left_lines, (0, 255, 0))    # Xanh lá
    right_lane_x = calculate_lane_x(right_lines, (255, 0, 0))  # Xanh dương

    # ============================================================
    # 6. LOGIC TÍNH TÂM ĐƯỜNG (Xử lý 3 trường hợp)
    # ============================================================
    
    if left_lane_x is not None and right_lane_x is not None:
        # CASE 1: Thấy cả 2 vạch - Hoàn hảo!
        x_line = (left_lane_x + right_lane_x) // 2
        cv2.circle(frame_debug, (left_lane_x, height - 10), 8, (0, 255, 0), -1)
        cv2.circle(frame_debug, (right_lane_x, height - 10), 8, (255, 0, 0), -1)
        cv2.putText(frame_debug, "BOTH LANES", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
    elif left_lane_x is not None:
        # CASE 2: Chỉ thấy TRÁI - Dự đoán tâm
        x_line = left_lane_x + (LANE_WIDTH_PIXELS // 2)
        cv2.circle(frame_debug, (left_lane_x, height - 10), 8, (0, 255, 0), -1)
        cv2.putText(frame_debug, "LEFT ONLY", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
    elif right_lane_x is not None:
        # CASE 3: Chỉ thấy PHẢI - Dự đoán tâm
        x_line = right_lane_x - (LANE_WIDTH_PIXELS // 2)
        cv2.circle(frame_debug, (right_lane_x, height - 10), 8, (255, 0, 0), -1)
        cv2.putText(frame_debug, "RIGHT ONLY", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
    else:
        # CASE 4: Mất cả 2 - Giữ nguyên hướng (đi thẳng)
        x_line = center_x
        cv2.putText(frame_debug, "NO LANE DETECTED", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # ============================================================
    # 7. TÍNH SAI SỐ VÀ VẼ DEBUG
    # ============================================================
    error = x_line - center_x
    
    # Vẽ đường tâm dự đoán (màu tím)
    cv2.line(frame_debug, (x_line, 0), (x_line, height), (255, 0, 255), 2)
    
    # Vẽ mũi tên chỉ hướng điều chỉnh
    cv2.arrowedLine(frame_debug, (center_x, height - 40), 
                    (x_line, height - 40), (0, 0, 255), 3, tipLength=0.3)
    
    # Hiển thị thông tin
    cv2.putText(frame_debug, f"Error: {error:+d}px", 
                (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    return error, x_line, center_x, frame_debug


def detect_line_black_adaptive(frame):
    """
    Phương pháp dự phòng: Sử dụng ADAPTIVE THRESHOLD cho vạch đen
    Phù hợp khi ánh sáng không đều
    """
    height, width = frame.shape[:2]
    center_x = width // 2
    
    frame_debug = frame.copy()
    
    # Chuyển sang grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Làm mờ
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # ADAPTIVE THRESHOLD - Tự động điều chỉnh theo vùng sáng tối
    # THRESH_BINARY_INV: Vạch đen thành trắng
    thresh = cv2.adaptiveThreshold(
        blur, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        blockSize=15,  # Kích thước vùng xét
        C=5            # Hằng số trừ đi
    )
    
    # ROI - Chỉ xét nửa dưới ảnh
    roi_top = height // 2
    thresh[:roi_top, :] = 0
    
    # Tìm contours (đường viền)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Lọc contours theo diện tích (loại nhiễu nhỏ)
    valid_contours = [c for c in contours if cv2.contourArea(c) > 100]
    
    if len(valid_contours) >= 2:
        # Sắp xếp theo vị trí x
        valid_contours = sorted(valid_contours, key=lambda c: cv2.boundingRect(c)[0])
        
        # Lấy 2 contours ngoài cùng (vạch trái và phải)
        left_contour = valid_contours[0]
        right_contour = valid_contours[-1]
        
        # Tính tâm của mỗi vạch
        M_left = cv2.moments(left_contour)
        M_right = cv2.moments(right_contour)
        
        if M_left['m00'] > 0 and M_right['m00'] > 0:
            left_x = int(M_left['m10'] / M_left['m00'])
            right_x = int(M_right['m10'] / M_right['m00'])
            
            x_line = (left_x + right_x) // 2
            
            cv2.drawContours(frame_debug, [left_contour], -1, (0, 255, 0), 2)
            cv2.drawContours(frame_debug, [right_contour], -1, (255, 0, 0), 2)
        else:
            x_line = center_x
    elif valid_contours:
        # Chỉ thấy 1 vạch
        M = cv2.moments(valid_contours[0])
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            # Dự đoán tâm (giả sử vạch ở bên trái)
            x_line = cx + 360  # Nửa lane width
            cv2.drawContours(frame_debug, [valid_contours[0]], -1, (0, 255, 0), 2)
        else:
            x_line = center_x
    else:
        x_line = center_x
    
    error = x_line - center_x
    
    # Vẽ debug
    cv2.line(frame_debug, (center_x, 0), (center_x, height), (0, 255, 255), 2)
    cv2.line(frame_debug, (x_line, 0), (x_line, height), (255, 0, 255), 2)
    cv2.arrowedLine(frame_debug, (center_x, height - 50), 
                    (x_line, height - 50), (0, 0, 255), 3)
    cv2.putText(frame_debug, f"Error: {error:+d}", 
                (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return error, x_line, center_x, frame_debug


# ============================================================
# CALIBRATION HELPER FUNCTION
# ============================================================
def calibrate_lane_width(frame, show_result=True):
    """
    Công cụ hỗ trợ CALIBRATE độ rộng lane (38cm) thành pixels
    
    Cách dùng:
    1. Đặt xe ở giữa lane
    2. Chụp ảnh test
    3. Chạy hàm này
    4. Click vào 2 vạch trái/phải tại đáy ảnh
    5. Nhận giá trị LANE_WIDTH_PIXELS
    """
    height, width = frame.shape[:2]
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_inv = cv2.bitwise_not(gray)
    edges = cv2.Canny(gray_inv, 30, 100)
    
    # Chỉ xét đáy ảnh (20% cuối)
    edges[:int(height * 0.8), :] = 0
    
    # Tìm đường thẳng
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 20, minLineLength=30, maxLineGap=15)
    
    frame_calib = frame.copy()
    
    if lines is not None:
        # Vẽ tất cả lines tìm được
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame_calib, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Tìm 2 x tại đáy ảnh (min và max)
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
                     (right_x, height - 30), (255, 255, 0), 2)
            
            cv2.putText(frame_calib, f"Lane Width: {lane_width_pixels}px (38cm)", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            print(f"\n{'='*60}")
            print(f"CALIBRATION RESULT:")
            print(f"  Lane Width (Real):  38 cm")
            print(f"  Lane Width (Pixel): {lane_width_pixels} px")
            print(f"  Scale Factor:       {38 / lane_width_pixels:.4f} cm/px")
            print(f"{'='*60}\n")
            
            if show_result:
                cv2.imshow("Calibration", frame_calib)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            return lane_width_pixels
    
    print("❌ Không tìm thấy 2 vạch để calibrate!")
    return None