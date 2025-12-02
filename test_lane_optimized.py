"""
Test Lane Detection - OPTIMIZED VERSION
Tests both Hough-based and Adaptive Threshold methods
Includes calibration tool
"""

import cv2
import os
import sys
import yaml

# Thêm đường dẫn
sys.path.append(os.getcwd())

try:
    # Import cả 2 phương pháp
    from perception.lane_detector import (
        detect_line, 
        detect_line_black_adaptive,
        calibrate_lane_width
    )
    from utils.config_loader import load_config
    print("✅ Import thành công!")
except ImportError as e:
    print(f"❌ Lỗi Import: {e}")
    print("Hãy chắc chắn bạn đã copy file lane_detector_optimized.py vào thư mục perception/")
    sys.exit(1)


# ============================================================
# BƯỚC 1: CALIBRATION (Chạy 1 lần đầu tiên)
# ============================================================
def run_calibration(test_file='test_full_hd.jpg'):
    """
    Chạy calibration để xác định LANE_WIDTH_PIXELS
    """
    print("\n" + "="*70)
    print("BƯỚC 1: CALIBRATION - Đo độ rộng lane thành pixels")
    print("="*70)
    
    if not os.path.exists(test_file):
        print(f"❌ Không tìm thấy file: {test_file}")
        return None
    
    frame = cv2.imread(test_file)
    if frame is None:
        print(f"❌ Không đọc được ảnh: {test_file}")
        return None
    
    # Resize về 640x480 (chuẩn của robot)
    frame = cv2.resize(frame, (640, 480))
    
    print("\n📸 Đang phân tích ảnh để tìm lane width...")
    lane_width_px = calibrate_lane_width(frame, show_result=True)
    
    if lane_width_px:
        print(f"\n✅ KẾT QUẢ CALIBRATION:")
        print(f"   Thêm dòng này vào lane_detector_optimized.py:")
        print(f"   LANE_WIDTH_PIXELS = {lane_width_px}")
        print(f"\n   Hoặc cập nhật trong config YAML:")
        print(f"   lane_width_pixels: {lane_width_px}")
    
    return lane_width_px


# ============================================================
# BƯỚC 2: TEST CẢ 2 PHƯƠNG PHÁP
# ============================================================
def run_tests(test_files, lane_config=None):
    """
    Test cả 2 phương pháp: Hough Transform và Adaptive Threshold
    """
    print("\n" + "="*70)
    print("BƯỚC 2: SO SÁNH 2 PHƯƠNG PHÁP LANE DETECTION")
    print("="*70)
    print(f"{'FILENAME':<25} | {'METHOD':<20} | {'ERROR':<8} | {'ACTION'}")
    print("-" * 70)
    
    for filename in test_files:
        if not os.path.exists(filename):
            print(f"{filename:<25} | ⚠️ File không tồn tại")
            continue
        
        frame = cv2.imread(filename)
        if frame is None:
            print(f"{filename:<25} | ❌ Lỗi đọc file")
            continue
        

        
        # ====================================================
        # METHOD 1: Hough Transform (Phương pháp chính)
        # ====================================================
        error_hough, x_line_hough, center_x, debug_hough = detect_line(frame, config=lane_config)
        
        if error_hough > 20:
            action_hough = "Rẽ PHẢI  (->)"
        elif error_hough < -20:
            action_hough = "Rẽ TRÁI  (<-)"
        else:
            action_hough = "Đi THẲNG (^)"
        
        print(f"{filename:<25} | {'Hough Transform':<20} | {error_hough:<8} | {action_hough}")
        
        # Lưu ảnh debug
        cv2.imwrite(f"debug_hough_{filename}", debug_hough)
        
        # ====================================================
        # METHOD 2: Adaptive Threshold (Phương pháp dự phòng)
        # ====================================================
        error_adaptive, x_line_adaptive, _, debug_adaptive = detect_line_black_adaptive(frame)
        
        if error_adaptive > 20:
            action_adaptive = "Rẽ PHẢI  (->)"
        elif error_adaptive < -20:
            action_adaptive = "Rẽ TRÁI  (<-)"
        else:
            action_adaptive = "Đi THẲNG (^)"
        
        print(f"{'':<25} | {'Adaptive Threshold':<20} | {error_adaptive:<8} | {action_adaptive}")
        
        # Lưu ảnh debug
        cv2.imwrite(f"debug_adaptive_{filename}", debug_adaptive)
        
        # So sánh kết quả
        diff = abs(error_hough - error_adaptive)
        if diff > 50:
            print(f"{'':<25} | ⚠️  Chênh lệch lớn giữa 2 phương pháp: {diff}px")
        
        print("-" * 70)


# ============================================================
# BƯỚC 3: TEST REAL-TIME (Nếu có camera)
# ============================================================
def test_realtime_camera():
    """
    Test với camera thực (Picamera2)
    """
    try:
        from picamera2 import Picamera2
        import time
        
        print("\n" + "="*70)
        print("BƯỚC 3: TEST REAL-TIME VỚI CAMERA")
        print("="*70)
        print("Nhấn 'q' để thoát | 'c' để chụp ảnh test | 's' để chuyển phương pháp")
        
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"size": (1640, 1232), "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()
        
        time.sleep(2)  # Warm-up
        
        method = 'hough'  # 'hough' hoặc 'adaptive'
        frame_count = 0
        
        # ... (đoạn code phía trên giữ nguyên)

        while True:
            # Capture frame
            frame_rgb = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frame_bgr = cv2.resize(frame_bgr, (640, 480))
            # Detect lane
            if method == 'hough':
                error, x_line, center_x, debug_frame = detect_line(frame_bgr)
                # ... (giữ nguyên phần putText)
            else:
                error, x_line, center_x, debug_frame = detect_line_black_adaptive(frame_bgr)
                # ... (giữ nguyên phần putText)
            
            # --- PHẦN SỬA ĐỔI: TẮT HIỂN THỊ, CHUYỂN SANG LƯU FILE ---
            
            # 1. Tắt hiển thị (Gây lỗi)
            # cv2.imshow("Lane Detection - Real-time", debug_frame)
            
            # 2. Tắt chờ phím (Vì không có cửa sổ để focus)
            # key = cv2.waitKey(1) & 0xFF
            
            # 3. Thay thế bằng logic lưu ảnh tự động
            print(f"\rProcessing Frame: {frame_count} | Error: {error}", end="")
            
            # Lưu ảnh mỗi 10 frame một lần để kiểm tra (tránh lưu quá nhiều)
            if frame_count % 10 == 0:
                filename = f"debug_stream_{frame_count}.jpg"
                cv2.imwrite(filename, debug_frame)
                
            # Tự động dừng sau 50 frame (để bạn không phải kẹt trong vòng lặp)
            if frame_count >= 50:
                print("\n✅ Đã chạy xong 50 frames test. Dừng lại.")
                break
            
            # Giả lập delay nhỏ để không chiếm hết CPU
            time.sleep(0.05)
            
            frame_count += 1
        
        picam2.stop()
        # ... (phần còn lại giữ nguyên)
        picam2.stop()
        picam2.close()
        cv2.destroyAllWindows()
        
        print(f"✅ Đã test {frame_count} frames")
        
    except ImportError:
        print("⚠️  Không tìm thấy Picamera2. Bỏ qua test real-time.")
    except Exception as e:
        print(f"❌ Lỗi: {e}")


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("CHƯƠNG TRÌNH TEST LANE DETECTION - PHIÊN BẢN TỐI ƯU")
    print("Tối ưu cho: Vạch ĐEN trên nền SÁNG | Lane 38cm | Camera 640x480")
    print("="*70)
    
    # Load config
    try:
        config_full = load_config('config/hardware_config.yaml')
        lane_config = config_full.get('ai', {}).get('lane_detection', {})
        print("✅ Đã tải config từ hardware_config.yaml")
    except:
        print("⚠️  Sử dụng config mặc định")
        lane_config = None
    
    # Danh sách ảnh test
    test_files = [
        'test_full_hd.jpg',
        'road_curve_left.jpg', 
        'road_curve_right.jpg'
    ]
    
    # Menu
    print("\nChọn chế độ test:")
    print("  1. CALIBRATION - Đo độ rộng lane (Chạy 1 lần đầu)")
    print("  2. TEST ẢNH TĨNH - So sánh 2 phương pháp")
    print("  3. TEST REAL-TIME - Camera thực (Picamera2)")
    print("  4. TẤT CẢ (Khuyến nghị)")
    
    choice = input("\nNhập lựa chọn (1-4): ").strip()
    
    if choice == '1':
        run_calibration('test_full_hd.jpg')
    
    elif choice == '2':
        run_tests(test_files, lane_config)
        print("\n✅ Hoàn tất! Kiểm tra các file 'debug_*.jpg'")
    
    elif choice == '3':
        test_realtime_camera()
    
    elif choice == '4':
        # Chạy tất cả
        print("\n🚀 Chạy đầy đủ quy trình...\n")
        
        # Step 1: Calibration
        lane_width_px = run_calibration('test_full_hd.jpg')
        
        input("\nNhấn Enter để tiếp tục test ảnh tĩnh...")
        
        # Step 2: Test static images
        run_tests(test_files, lane_config)
        
        print("\n✅ Đã hoàn thành test ảnh tĩnh!")
        
        # Step 3: Ask for real-time test
        do_realtime = input("\nBạn có muốn test real-time với camera không? (y/n): ")
        if do_realtime.lower() == 'y':
            test_realtime_camera()
    
    else:
        print("❌ Lựa chọn không hợp lệ!")
    
    print("\n" + "="*70)
    print("KẾT THÚC CHƯƠNG TRÌNH")
    print("="*70)