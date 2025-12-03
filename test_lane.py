
import cv2
import os
import sys
import yaml

# Thêm đường dẫn để import các module của dự án
sys.path.append(os.getcwd())

try:
    from perception.lane_detector import detect_line
    from utils.config_loader import load_config
except ImportError as e:
    print(f"❌ Lỗi Import: {e}")
    print("Hãy chắc chắn bạn đang chạy lệnh từ thư mục ~/logisticsbot")
    sys.exit(1)

# 1. Tải cấu hình (QUAN TRỌNG: Để áp dụng các tham số bạn vừa tune)
try:
    config_full = load_config('config/hardware_config.yaml')
    # Lấy đúng phần config cho lane detection
    lane_config = config_full.get('ai', {}).get('lane_detection', {})
    print("✅ Đã tải cấu hình từ hardware_config.yaml")
    print(f"   -> Hough Threshold: {lane_config.get('hough_threshold')}")
    print(f"   -> Max Line Gap:    {lane_config.get('max_line_gap')}")
except Exception as e:
    print(f"⚠️ Không đọc được config: {e}")
    print("-> Sẽ sử dụng tham số mặc định trong code (có thể không chính xác)")
    lane_config = None

# Danh sách ảnh cần test (Ảnh bạn đã tạo trước đó)
test_files = ['test_full_hd.jpg', 'road_curve_left.jpg', 'road_curve_right.jpg']

print("\n" + "="*60)
print("TEST LANE DETECTION VỚI ẢNH TĨNH")
print("="*60)
print(f"{'FILENAME':<20} | {'ERROR':<10} | {'ACTION'}")
print("-" * 60)

# ... (Phần đầu giữ nguyên) ...

print("-" * 60)

for filename in test_files:
    if not os.path.exists(filename):
        print(f"{filename:<20} | ⚠️ File không tồn tại")
        continue
        
    # 2. Đọc ảnh
    frame = cv2.imread(filename)
    if frame is None:
        print(f"{filename:<20} | ❌ Lỗi đọc file ảnh")
        continue

    # === BỔ SUNG: Resize về chuẩn 1640x1232 để khớp thông số Lane Detector ===
    frame = cv2.resize(frame, (1640, 1232))
    # ======================================================================

    # 3. Chạy thuật toán phát hiện làn đường
    error, x_line, center_x, debug_frame = detect_line(frame, config=lane_config)
    
    # 4. Đánh giá hành động
    if error > 20:
        action = "Rẽ PHẢI  (->)"
    elif error < -20:
        action = "Rẽ TRÁI  (<-)"
    else:
        action = "Đi THẲNG (^)"

    # 5. In kết quả
    print(f"{filename:<20} | {error:<10} | {action}")
    
    # 6. Lưu ảnh kết quả debug
    out_name = f"debug_{filename}"
    cv2.imwrite(out_name, debug_frame)

print("-" * 60)
print("✅ Hoàn tất! Hãy kiểm tra các file 'debug_*.jpg'.")