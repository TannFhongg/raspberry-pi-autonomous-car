"""
Fixed Calibration - Chỉ đo 2 vạch lane chính, bỏ qua nhiễu
"""

import cv2
import numpy as np


def calibrate_lane_width_fixed(frame, show_result=False):
    """
    Calibration tool - FIXED VERSION
    Chỉ đo 2 vạch lane chính, bỏ qua nhiễu từ background
    """
    # Resize về 640x480 nếu cần
    if frame.shape[1] != 640:
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
        print(f"[INFO] Resized to 640x480")
    
    height, width = frame.shape[:2]
    center_x = width // 2
    
    # ============================================
    # BƯỚC 1: Tiền xử lý
    # ============================================
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_inv = cv2.bitwise_not(gray)
    
    # Tăng cường tương phản
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_inv)
    
    # Canny
    edges = cv2.Canny(enhanced, 40, 120)
    
    # ============================================
    # BƯỚC 2: ROI CHẶT CHẼ - CHỈ VÙNG LANE
    # ============================================
    # Chỉ xét 30% đáy ảnh (rất gần xe)
    roi_top = int(height * 0.70)  # Từ 70% xuống 100%
    roi_bottom = height
    
    # Thu hẹp 2 bên (40%-60% thay vì 0%-100%)
    roi_left = int(width * 0.20)   # Bỏ 20% bên trái
    roi_right = int(width * 0.80)  # Bỏ 20% bên phải
    
    # Tạo mask ROI hình chữ nhật
    mask = np.zeros_like(edges)
    cv2.rectangle(mask, (roi_left, roi_top), (roi_right, roi_bottom), 255, -1)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    frame_calib = frame.copy()
    cv2.rectangle(frame_calib, (roi_left, roi_top), (roi_right, roi_bottom), (255, 0, 0), 2)
    
    # ============================================
    # BƯỚC 3: Hough Transform
    # ============================================
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 25, minLineLength=40, maxLineGap=20)
    
    if lines is None:
        print("❌ Không tìm thấy đường thẳng nào!")
        return None
    
    print(f"[INFO] Tìm thấy {len(lines)} đường thẳng trong ROI")
    
    # ============================================
    # BƯỚC 4: Phân loại TRÁI/PHẢI
    # ============================================
    left_lines = []
    right_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        if abs(x2 - x1) < 1:
            continue
        
        slope = (y2 - y1) / (x2 - x1)
        
        # Lọc slope
        if abs(slope) < 0.5:
            continue
        
        mid_x = (x1 + x2) / 2
        
        # Phân loại
        if slope < 0 and mid_x < center_x:
            left_lines.append((x1, y1, x2, y2, slope))
            cv2.line(frame_calib, (x1, y1), (x2, y2), (0, 255, 0), 2)
        elif slope > 0 and mid_x > center_x:
            right_lines.append((x1, y1, x2, y2, slope))
            cv2.line(frame_calib, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    print(f"[INFO] Left lines: {len(left_lines)}, Right lines: {len(right_lines)}")
    
    # ============================================
    # BƯỚC 5: Tính toán vị trí vạch tại ĐÁY ẢNH
    # ============================================
    def calculate_x_at_bottom(lines):
        if not lines:
            return None
        
        x_bottoms = []
        for x1, y1, x2, y2, slope in lines:
            x_bottom = x1 + (height - y1) / slope
            
            # Kiểm tra hợp lý
            if 0 <= x_bottom <= width:
                x_bottoms.append(x_bottom)
        
        if x_bottoms:
            return int(np.median(x_bottoms))  # Dùng MEDIAN thay vì MEAN
        return None
    
    left_x = calculate_x_at_bottom(left_lines)
    right_x = calculate_x_at_bottom(right_lines)
    
    # ============================================
    # BƯỚC 6: Kiểm tra kết quả
    # ============================================
    if left_x is None or right_x is None:
        print("❌ Không tìm thấy cả 2 vạch!")
        print(f"   Left X: {left_x}, Right X: {right_x}")
        print("💡 Thử:")
        print("   - Đặt xe GẦN HƠN với lane")
        print("   - Đảm bảo 2 vạch nằm trong ROI (hình chữ nhật xanh)")
        return None
    
    # Tính lane width
    lane_width_pixels = right_x - left_x
    
    # Kiểm tra tính hợp lý (25cm không thể > 400px tại 640x480)
    if lane_width_pixels > 400 or lane_width_pixels < 100:
        print(f"⚠️  Kết quả không hợp lý: {lane_width_pixels}px")
        print(f"   Left X: {left_x}, Right X: {right_x}")
        print("💡 Có thể:")
        print("   - Xe quá xa lane (lane quá lớn)")
        print("   - Xe quá gần lane (lane quá nhỏ)")
        print("   - Detect sai vạch")
        return None
    
    # Vẽ kết quả
    cv2.circle(frame_calib, (left_x, height - 10), 12, (0, 255, 0), -1)
    cv2.circle(frame_calib, (right_x, height - 10), 12, (255, 0, 0), -1)
    cv2.line(frame_calib, (left_x, height - 30), (right_x, height - 30), (255, 255, 0), 4)
    
    cv2.putText(frame_calib, f"Lane: {lane_width_pixels}px = 25cm", 
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame_calib, f"Left: {left_x}px | Right: {right_x}px", 
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Lưu ảnh
    cv2.imwrite("calibration_result_fixed.jpg", frame_calib)
    
    # In kết quả
    print(f"\n{'='*60}")
    print(f"✅ CALIBRATION THÀNH CÔNG:")
    print(f"  Lane Width (Real):  25 cm")
    print(f"  Lane Width (Pixel): {lane_width_pixels} px")
    print(f"  Scale Factor:       {25 / lane_width_pixels:.4f} cm/px")
    print(f"  Left X:  {left_x} px")
    print(f"  Right X: {right_x} px")
    print(f"  📸 Đã lưu: calibration_result_fixed.jpg")
    print(f"{'='*60}\n")
    print(f"⚠️  CẬP NHẬT NGAY:")
    print(f"  Sửa dòng 53 trong lane_detector.py:")
    print(f"  LANE_WIDTH_PIXELS = {lane_width_pixels}")
    print(f"{'='*60}\n")
    
    return lane_width_pixels


# ============================================
# TEST SCRIPT
# ============================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 calibrate_fixed.py <image_file>")
        print("Example: python3 calibrate_fixed.py test_full_hd.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    print(f"📸 Đọc ảnh: {image_path}")
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"❌ Không đọc được ảnh: {image_path}")
        sys.exit(1)
    
    print(f"   Kích thước gốc: {frame.shape[1]}x{frame.shape[0]}")
    
    # Chạy calibration
    result = calibrate_lane_width_fixed(frame, show_result=False)
    
    if result:
        print(f"✅ Thành công! Lane width = {result} pixels")
    else:
        print(f"❌ Calibration thất bại. Xem hướng dẫn ở trên.")
