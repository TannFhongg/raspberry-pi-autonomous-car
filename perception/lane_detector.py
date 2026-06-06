"""
Lane Detection Module - FIXED for 1640x1232 → 640x480 resize
OPTIMIZED for BLACK LINES on WHITE BACKGROUND
Designed for: 38cm lane width, 15cm robot width, black tape lines
Camera: Raspberry Pi Camera Module 2

CHANGELOG:
- FIX: Thêm tham số debug=False vào detect_line() và detect_line_black_adaptive()
  Robot controller gọi detect_line(..., debug=False) nhưng signature cũ không có param này
  → TypeError crash → auto loop thoát ngay lập tức → không có debug frame nào được lưu
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def detect_line(frame, config=None, debug=False):
    """
    Phát hiện vạch KẺ ĐEN trên nền TRẮNG (bìa trắng)
    ✅ OPTIMIZED: Nhận RAW YUV420 từ camera_manager, lấy trực tiếp kênh Y (grayscale)
    ✅ FIX: Thêm tham số debug=False để tương thích với robot_controller.py
    
    Args:
        frame:  RAW YUV420 array (720, 640) từ camera_manager, hoặc BGR (480, 640, 3)
        config: Cấu hình lane detection từ hardware_config.yaml
        debug:  (Deprecated) Không dùng nữa, giữ để backward compatible
    
    Returns:
        (error, x_line, center_x, None)
    
    Thông số thực tế:
    - Lane width: 38cm
    - Robot width: 15cm
    - Input: RAW YUV420 640x720 từ camera_manager (ISP hardware output)
    - Line color: BLACK on WHITE background
    """
    
    # ============================================================
    # BƯỚC 0: XỬ LÝ FORMAT - LẤY KÊNH Y (GRAYSCALE) TỪ YUV420
    # ============================================================
    # ✅ OPTIMIZED: YUV420 format từ ISP
    # - Shape: (H*3//2, W) = (720, 640) cho 640x480 resolution
    # - Layout: Y plane (480×640) + U plane (240×320) + V plane (240×320)
    # - Kênh Y = Grayscale MIỄN PHÍ từ ISP (không tốn CPU)
    
    height, width = frame.shape[:2]
    
    # Detect format và extract grayscale
    if len(frame.shape) == 2:
        # YUV420 planar format: shape = (H*3//2, W)
        if height == 720 and width == 640:
            # Extract Y channel (first 480 rows)
            gray = frame[:480, :].copy()
            logger.debug("✅ OPTIMIZED: Extracted Y channel from YUV420 (CPU cost = 0)")
        else:
            # Unknown 2D format - assume grayscale
            gray = frame
            logger.warning(f"Unexpected 2D frame size: {frame.shape}, assuming grayscale")
    elif len(frame.shape) == 3 and frame.shape[2] == 3:
        # BGR format (fallback for testing or if camera_manager changed)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        logger.warning("⚠️ Received BGR frame, converting to grayscale (CPU overhead!)")
    else:
        # Unknown format - log error
        logger.error(f"Unexpected frame format: shape={frame.shape}")
        # Return safe defaults
        return 999, width // 2, width // 2, frame
    
    # Đảm bảo kích thước đúng 640x480
    if gray.shape != (480, 640):
        gray = cv2.resize(gray, (640, 480), interpolation=cv2.INTER_AREA)
        logger.warning(f"Frame không phải 640x480, đã resize: {gray.shape}")
    
    height, width = 480, 640  # Cố định
    center_x = width // 2
    
    # ============================================================
    # Cấu hình mặc định - TUNED cho vạch đen trên nền trắng
    # ============================================================
    if config is None:
        config = {
            'roi_top_ratio': 0.6,
            'roi_bottom_ratio': 1.0,
            'canny_low': 80,
            'canny_high': 185,
            'hough_threshold': 45,
            'min_line_length': 60,
            'max_line_gap': 30,
            'blur_kernel': 7,
        }

    # ============================================================
    # LANE WIDTH PIXELS - ĐÃ CALIBRATE CHO 640x480
    # ============================================================
    LANE_WIDTH_PIXELS = 457  # ⚠️ GIÁ TRỊ ƯỚC TÍNH - PHẢI CALIBRATE!

    # ============================================================
    # 1. TIỀN XỬ LÝ ẢNH - CHO NỀN TRẮNG, VẠCH ĐEN
    # ============================================================
    
    # ĐẢO NGƯỢC: Vạch đen → trắng (Canny hoạt động tốt hơn)
    gray_inverted = cv2.bitwise_not(gray)
    
    # Làm mờ nhẹ (nền trắng ít nhiễu hơn nền nhà)
    blur = cv2.GaussianBlur(gray_inverted, (config['blur_kernel'], config['blur_kernel']), 0)
    
    # TĂNG CƯỜNG TƯƠNG PHẢN
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(blur)
    
    # Canny edge detection
    edges = cv2.Canny(enhanced, config['canny_low'], config['canny_high'])
    
    # ============================================================
    # 2. XÁC ĐỊNH ROI - HÌNH THANG RỘNG HƠN
    # ============================================================
    roi_top = int(height * config['roi_top_ratio'])
    roi_bottom = int(height * config['roi_bottom_ratio'])
    
    roi_vertices = np.array([[
        (0, roi_bottom),
        (int(width * 0.2), roi_top),
        (int(width * 0.8), roi_top),
        (width, roi_bottom)
    ]], dtype=np.int32)
    
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

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
    # 4. PHÂN LOẠI VẠCH TRÁI/PHẢI
    # ============================================================
    left_lines = []
    right_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            if abs(x2 - x1) < 1:
                continue
            
            slope = (y2 - y1) / (x2 - x1)
            
            if abs(slope) < 0.5:
                continue
            
            mid_x = (x1 + x2) / 2
            
            if slope < -0.5 and mid_x < center_x:
                left_lines.append((x1, y1, x2, y2, slope))
            elif slope > 0.5 and mid_x > center_x:
                right_lines.append((x1, y1, x2, y2, slope))
    
    # ============================================================
    # 5. TÍNH TOÁN VỊ TRÍ VẠCH (Extrapolate về đáy ảnh)
    # ============================================================
    left_lane_x = None
    right_lane_x = None
    
    def calculate_lane_x(lines):
        """Tính tọa độ x tại đáy ảnh từ danh sách lines"""
        if not lines:
            return None
        
        x_bottoms = []
        
        for x1, y1, x2, y2, slope in lines:
            x_bottom = x1 + (height - y1) / slope
            
            if 0 <= x_bottom <= width:
                x_bottoms.append(x_bottom)
        
        if x_bottoms:
            return int(np.median(x_bottoms))
        return None
    
    left_lane_x = calculate_lane_x(left_lines)
    right_lane_x = calculate_lane_x(right_lines)

    # ============================================================
    # 6. LOGIC TÍNH TÂM ĐƯỜNG (3 trường hợp)
    # ============================================================
    
    lane_status = "UNKNOWN"
    
    if left_lane_x is not None and right_lane_x is not None:
        x_line = (left_lane_x + right_lane_x) // 2
        lane_status = "BOTH_LANES"
        
    elif left_lane_x is not None:
        x_line = left_lane_x + (LANE_WIDTH_PIXELS // 2)
        lane_status = "LEFT_ONLY"
        
    elif right_lane_x is not None:
        x_line = right_lane_x - (LANE_WIDTH_PIXELS // 2)
        lane_status = "RIGHT_ONLY"
        
    else:
        x_line = center_x
        lane_status = "NO_LANE"

    # ============================================================
    # 7. TÍNH SAI SỐ
    # ============================================================
    CAMERA_OFFSET = -15
    if lane_status == "NO_LANE":
        error = 999
    else:
        error = x_line - center_x + CAMERA_OFFSET
    
    return error, x_line, center_x, None


def detect_line_black_adaptive(frame, debug=False):
    """
    Phương pháp dự phòng: ADAPTIVE THRESHOLD
    ✅ FIX: Thêm tham số debug=False để tương thích
    ✅ OPTIMIZED: Nhận RAW YUV420, lấy trực tiếp kênh Y
    """
    height, width = frame.shape[:2]
    
    if len(frame.shape) == 2:
        if height == 720 and width == 640:
            gray = frame[:480, :].copy()
        else:
            gray = frame
    elif len(frame.shape) == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        logger.warning("⚠️ Adaptive: Received BGR, converting (CPU overhead!)")
    else:
        logger.error(f"Unexpected frame format in adaptive: shape={frame.shape}")
        return 999, width // 2, width // 2, frame
    
    if gray.shape != (480, 640):
        gray = cv2.resize(gray, (640, 480), interpolation=cv2.INTER_AREA)
    
    height, width = 480, 640
    center_x = width // 2
    LANE_WIDTH_PIXELS = 457
    
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    
    thresh = cv2.adaptiveThreshold(
        blur, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        blockSize=21,
        C=8
    )
    
    roi_top = int(height * 0.35)
    thresh[:roi_top, :] = 0
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) > 200]
    
    if len(valid_contours) >= 2:
        valid_contours = sorted(valid_contours, key=lambda c: cv2.boundingRect(c)[0])
        
        left_contour = valid_contours[0]
        right_contour = valid_contours[-1]
        
        M_left = cv2.moments(left_contour)
        M_right = cv2.moments(right_contour)
        
        if M_left['m00'] > 0 and M_right['m00'] > 0:
            left_x = int(M_left['m10'] / M_left['m00'])
            right_x = int(M_right['m10'] / M_right['m00'])
            x_line = (left_x + right_x) // 2
        else:
            x_line = center_x
            
    elif valid_contours:
        M = cv2.moments(valid_contours[0])
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            if cx < center_x:
                x_line = cx + (LANE_WIDTH_PIXELS // 2)
            else:
                x_line = cx - (LANE_WIDTH_PIXELS // 2)
        else:
            x_line = center_x
    else:
        x_line = center_x
    
    error = x_line - center_x
    
    return error, x_line, center_x, None


def calibrate_lane_width(frame, show_result=False):
    """
    Calibration tool - Đo 38cm lane thành pixels
    ✅ OPTIMIZED: Nhận RAW YUV420, lấy trực tiếp kênh Y
    """
    height, width = frame.shape[:2]
    
    if len(frame.shape) == 2:
        if height == 720 and width == 640:
            gray = frame[:480, :].copy()
        else:
            gray = frame
    elif len(frame.shape) == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        logger.warning("⚠️ Calibrate: Received BGR, converting (CPU overhead!)")
    else:
        logger.error(f"Unexpected frame format in calibrate: shape={frame.shape}")
        return None
    
    if gray.shape != (480, 640):
        gray = cv2.resize(gray, (640, 480), interpolation=cv2.INTER_AREA)
    
    height, width = 480, 640
    
    gray_inv = cv2.bitwise_not(gray)
    edges = cv2.Canny(gray_inv, 40, 120)
    
    edges[:int(height * 0.8), :] = 0
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 25, minLineLength=35, maxLineGap=20)
    
    if lines is not None:
        x_coords = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            x_coords.extend([x1, x2])
        
        if x_coords:
            left_x = min(x_coords)
            right_x = max(x_coords)
            lane_width_pixels = right_x - left_x
            
            print(f"\n{'='*60}")
            print(f"✅ CALIBRATION THÀNH CÔNG:")
            print(f"  Lane Width (Real):  38 cm")
            print(f"  Lane Width (Pixel): {lane_width_pixels} px")
            print(f"  Scale Factor:       {38 / lane_width_pixels:.4f} cm/px")
            print(f"{'='*60}\n")
            print(f"⚠️  CẬP NHẬT NGAY:")
            print(f"  Sửa LANE_WIDTH_PIXELS trong lane_detector.py:")
            print(f"  LANE_WIDTH_PIXELS = {lane_width_pixels}")
            print(f"{'='*60}\n")
            
            return lane_width_pixels
    
    print("❌ Không tìm thấy 2 vạch để calibrate!")
    return None