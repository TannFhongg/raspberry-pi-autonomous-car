"""
Lane Detection Module - resolution-aware frame processing
OPTIMIZED for BLACK LINES on WHITE BACKGROUND
Designed for: 38cm lane width, 15cm robot width, black tape lines
Camera: Raspberry Pi Camera Module 2
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


DEFAULT_LANE_CONFIG = {
    'roi_top_ratio': 0.6,
    'roi_bottom_ratio': 1.0,
    'roi_left_ratio': 0.1,
    'roi_right_ratio': 0.9,
    'canny_low': 80,
    'canny_high': 185,
    'hough_threshold': 45,
    'min_line_length': 60,
    'max_line_gap': 30,
    'blur_kernel': 7,
    'lane_width_pixels': 457,
    'camera_offset': -15,
}


def _finite_number(value, default, name):
    try:
        number = float(value)
    except (TypeError, ValueError):
        logger.warning(f"Invalid lane config {name}={value!r}; using {default}")
        return default

    if not np.isfinite(number):
        logger.warning(f"Non-finite lane config {name}={value!r}; using {default}")
        return default

    return number


def _clamp_float(value, default, minimum, maximum, name):
    number = _finite_number(value, default, name)
    return max(minimum, min(maximum, float(number)))


def _clamp_int(value, default, minimum, maximum, name):
    number = _finite_number(value, default, name)
    return max(minimum, min(maximum, int(round(number))))


def _normalize_blur_kernel(value, default=7):
    kernel = _clamp_int(value, default, 1, 99, 'blur_kernel')
    if kernel % 2 == 0:
        kernel += 1
    return kernel


def normalize_lane_config(config=None):
    """Return lane detection config with safe numeric values."""
    merged = DEFAULT_LANE_CONFIG.copy()
    if config:
        merged.update(config)

    normalized = {
        'roi_top_ratio': _clamp_float(
            merged.get('roi_top_ratio'),
            DEFAULT_LANE_CONFIG['roi_top_ratio'],
            0.0,
            0.99,
            'roi_top_ratio',
        ),
        'roi_bottom_ratio': _clamp_float(
            merged.get('roi_bottom_ratio'),
            DEFAULT_LANE_CONFIG['roi_bottom_ratio'],
            0.01,
            1.0,
            'roi_bottom_ratio',
        ),
        'roi_left_ratio': _clamp_float(
            merged.get('roi_left_ratio'),
            DEFAULT_LANE_CONFIG['roi_left_ratio'],
            0.0,
            0.99,
            'roi_left_ratio',
        ),
        'roi_right_ratio': _clamp_float(
            merged.get('roi_right_ratio'),
            DEFAULT_LANE_CONFIG['roi_right_ratio'],
            0.01,
            1.0,
            'roi_right_ratio',
        ),
        'canny_low': _clamp_int(
            merged.get('canny_low'), DEFAULT_LANE_CONFIG['canny_low'], 0, 1000, 'canny_low'
        ),
        'canny_high': _clamp_int(
            merged.get('canny_high'), DEFAULT_LANE_CONFIG['canny_high'], 0, 1000, 'canny_high'
        ),
        'hough_threshold': _clamp_int(
            merged.get('hough_threshold'),
            DEFAULT_LANE_CONFIG['hough_threshold'],
            1,
            1000,
            'hough_threshold',
        ),
        'min_line_length': _clamp_int(
            merged.get('min_line_length'),
            DEFAULT_LANE_CONFIG['min_line_length'],
            1,
            10000,
            'min_line_length',
        ),
        'max_line_gap': _clamp_int(
            merged.get('max_line_gap'),
            DEFAULT_LANE_CONFIG['max_line_gap'],
            0,
            10000,
            'max_line_gap',
        ),
        'blur_kernel': _normalize_blur_kernel(
            merged.get('blur_kernel'), DEFAULT_LANE_CONFIG['blur_kernel']
        ),
        'lane_width_pixels': _clamp_int(
            merged.get('lane_width_pixels'),
            DEFAULT_LANE_CONFIG['lane_width_pixels'],
            1,
            10000,
            'lane_width_pixels',
        ),
        'camera_offset': _clamp_int(
            merged.get('camera_offset'),
            DEFAULT_LANE_CONFIG['camera_offset'],
            -10000,
            10000,
            'camera_offset',
        ),
    }

    if normalized['roi_bottom_ratio'] <= normalized['roi_top_ratio']:
        logger.warning(
            "Invalid ROI vertical range: top=%s bottom=%s; using defaults",
            normalized['roi_top_ratio'],
            normalized['roi_bottom_ratio'],
        )
        normalized['roi_top_ratio'] = DEFAULT_LANE_CONFIG['roi_top_ratio']
        normalized['roi_bottom_ratio'] = DEFAULT_LANE_CONFIG['roi_bottom_ratio']

    if normalized['roi_right_ratio'] <= normalized['roi_left_ratio']:
        logger.warning(
            "Invalid ROI horizontal range: left=%s right=%s; using defaults",
            normalized['roi_left_ratio'],
            normalized['roi_right_ratio'],
        )
        normalized['roi_left_ratio'] = DEFAULT_LANE_CONFIG['roi_left_ratio']
        normalized['roi_right_ratio'] = DEFAULT_LANE_CONFIG['roi_right_ratio']

    if normalized['canny_high'] < normalized['canny_low']:
        logger.warning(
            "Canny high threshold lower than low threshold; swapping %s/%s",
            normalized['canny_low'],
            normalized['canny_high'],
        )
        normalized['canny_low'], normalized['canny_high'] = (
            normalized['canny_high'],
            normalized['canny_low'],
        )

    return normalized


def _looks_like_yuv420_shape(height, width):
    if height <= 0 or width <= 0:
        return False
    if height % 3 != 0 or width % 2 != 0:
        return False

    image_height = (height * 2) // 3
    if image_height <= 0 or image_height % 2 != 0:
        return False

    aspect_ratio = image_height / float(width)
    return 0.55 <= aspect_ratio <= 1.8


def _extract_gray_frame(frame, debug=False, context="detect_line"):
    """
    Extract grayscale from YUV420/BGR/grayscale input without assuming resolution.
    """
    if frame is None or not hasattr(frame, "shape"):
        logger.error(f"Unexpected frame in {context}: {type(frame)!r}")
        return None, None

    if len(frame.shape) == 2:
        height, width = frame.shape[:2]
        if _looks_like_yuv420_shape(height, width):
            image_height = (height * 2) // 3
            gray = frame[:image_height, :].copy()
            frame_color = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420) if debug else None
            logger.debug(
                "Extracted Y channel from YUV420 frame: raw=%s gray=%s",
                frame.shape,
                gray.shape,
            )
            return gray, frame_color

        gray = frame.copy()
        frame_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if debug else None
        logger.debug("Using 2D grayscale frame: shape=%s", frame.shape)
        return gray, frame_color

    if len(frame.shape) == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_color = frame.copy() if debug else None
        logger.debug("Converted BGR frame to grayscale: shape=%s", frame.shape)
        return gray, frame_color

    logger.error(f"Unexpected frame format in {context}: shape={frame.shape}")
    return None, None


def detect_line(frame, config=None, debug=False):
    """
    Phát hiện vạch KẺ ĐEN trên nền TRẮNG (bìa trắng)
    ✅ OPTIMIZED: Nhận RAW YUV420 từ camera_manager, lấy trực tiếp kênh Y (grayscale)

    Thông số thực tế:
    - Lane width: 38cm
    - Robot width: 15cm
    - Input: RAW YUV420 từ camera_manager (ISP hardware output)
    - Line color: BLACK on WHITE background

    Args:
        frame: Input frame (YUV420 or BGR)
        config: Lane detection config dict
        debug: If True, return a clean BGR frame for diagnostic tools; no drawing is performed

    Returns:
        (error, x_line, center_x, debug_frame or None)
    """

    config = normalize_lane_config(config)
    gray, frame_color = _extract_gray_frame(frame, debug=debug, context="detect_line")
    if gray is None:
        return 999, 0, 0, None

    height, width = gray.shape[:2]
    center_x = width // 2

    # ============================================================
    # LANE WIDTH PIXELS - Đọc từ config/calibration
    # ============================================================
    lane_width_pixels = config['lane_width_pixels']

    # ============================================================
    # 1. TIỀN XỬ LÝ ẢNH - CHO NỀN TRẮNG, VẠCH ĐEN
    # ============================================================

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
    roi_left = int(width * config['roi_left_ratio'])
    roi_right = int(width * config['roi_right_ratio'])
    roi_vertices = np.array([[
        (0, roi_bottom),
        (roi_left, roi_top),
        (roi_right, roi_top),
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

    def calculate_lane_x(lines):
        """Tính tọa độ x tại đáy ảnh từ danh sách lines"""
        if not lines:
            return None

        x_bottoms = []
        for x1, y1, x2, y2, slope in lines:
            # Ngoại suy đến đáy ảnh: x_bottom = x1 + (height - y1) / slope
            x_bottom = x1 + (height - y1) / slope

            # Kiểm tra x_bottom có hợp lý không (trong khoảng 0 - width)
            if 0 <= x_bottom <= width:
                x_bottoms.append(x_bottom)

        if x_bottoms:
            # Lấy MEDIAN thay vì MEAN (chống outlier tốt hơn)
            return int(np.median(x_bottoms))
        return None

    left_lane_x = calculate_lane_x(left_lines)
    right_lane_x = calculate_lane_x(right_lines)

    # ============================================================
    # 6. LOGIC TÍNH TÂM ĐƯỜNG (3 trường hợp)
    # ============================================================

    lane_status = "UNKNOWN"

    if left_lane_x is not None and right_lane_x is not None:
        # CASE 1: Thấy cả 2 vạch - HOÀN HẢO
        x_line = (left_lane_x + right_lane_x) // 2
        lane_status = "BOTH_LANES"

    elif left_lane_x is not None:
        # CASE 2: Chỉ thấy TRÁI
        x_line = left_lane_x + (lane_width_pixels // 2)
        lane_status = "LEFT_ONLY"

    elif right_lane_x is not None:
        # CASE 3: Chỉ thấy PHẢI
        x_line = right_lane_x - (lane_width_pixels // 2)
        lane_status = "RIGHT_ONLY"

    else:
        # CASE 4: Mất cả 2
        x_line = center_x
        lane_status = "NO_LANE"

    # ============================================================
    # 7. TÍNH SAI SỐ
    # ============================================================
    camera_offset = config['camera_offset']  # Hiệu chỉnh nếu camera không đặt chính giữa robot
    if lane_status == "NO_LANE":
        error = 999  # ⚠️ QUAN TRỌNG: Gán cứng lỗi 999 khi mất line
    else:
        error = x_line - center_x + camera_offset  # Các trường hợp còn lại tính toán bình thường

    return error, x_line, center_x, frame_color


def detect_line_black_adaptive(frame, debug=False, config=None):
    """
    Phương pháp dự phòng: ADAPTIVE THRESHOLD
    Tốt hơn khi ánh sáng không đều hoặc Hough thất bại
    ✅ OPTIMIZED: Nhận RAW YUV420, lấy trực tiếp kênh Y

    Args:
        frame: Input frame (YUV420 or BGR)
        debug: If True, return debug frame with visualization (default: False)

    Returns:
        (error, x_line, center_x, debug_frame or None)
    """
    if isinstance(debug, dict) and config is None:
        config = debug
        debug = False

    config = normalize_lane_config(config)
    gray, frame_debug = _extract_gray_frame(frame, debug=debug, context="adaptive")
    if gray is None:
        return 999, 0, 0, None

    height, width = gray.shape[:2]
    center_x = width // 2
    lane_width_pixels = config['lane_width_pixels']

    # Làm mờ
    blur_kernel = config['blur_kernel']
    blur = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

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

        else:
            x_line = center_x

    elif valid_contours:
        # Chỉ thấy 1 vạch
        M = cv2.moments(valid_contours[0])
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])

            # Dự đoán: Nếu contour ở bên trái → thêm nửa lane width
            if cx < center_x:
                x_line = cx + (lane_width_pixels // 2)
            else:
                x_line = cx - (lane_width_pixels // 2)
        else:
            x_line = center_x
    else:
        x_line = center_x

    error = x_line - center_x

    return error, x_line, center_x, frame_debug


def calibrate_lane_width(frame, show_result=False):
    """
    Calibration tool - Đo 38cm lane thành pixels
    ✅ OPTIMIZED: Nhận RAW YUV420, lấy trực tiếp kênh Y
    Đã sửa: KHÔNG dùng cv2.imshow() (không có màn hình)
    """
    gray, _ = _extract_gray_frame(frame, debug=False, context="calibrate")
    if gray is None:
        return None

    height, width = gray.shape[:2]

    gray_inv = cv2.bitwise_not(gray)
    edges = cv2.Canny(gray_inv, 40, 120)

    # Chỉ xét 20% đáy ảnh
    edges[:int(height * 0.8), :] = 0

    # Tìm đường thẳng
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 25, minLineLength=35, maxLineGap=20)

    if lines is not None:
        # Tìm x min và max
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
            print(f"  config/hardware_config.yaml:")
            print(f"  ai.lane_detection.lane_width_pixels: {lane_width_pixels}")
            print(f"{'='*60}\n")

            return lane_width_pixels

    print("❌ Không tìm thấy 2 vạch để calibrate!")
    print("💡 Kiểm tra:")
    print("  - Xe có đang ở giữa lane không?")
    print("  - Vạch đen có rõ ràng trên nền trắng không?")
    print("  - Ánh sáng có đủ không?")
    return None
