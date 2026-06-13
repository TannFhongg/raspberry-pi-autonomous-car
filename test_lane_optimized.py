"""
Test Lane Detection - FIXED VERSION
- Hỗ trợ resolution 1640x1232 (tự động resize về 640x480)
- KHÔNG dùng cv2.imshow() (không có màn hình ngoài)
- Lưu tất cả kết quả ra file ảnh
"""
import numpy as np

import cv2
import os
import sys
import yaml

sys.path.append(os.getcwd())

try:
    from perception.lane_detector import (
        detect_line, 
        detect_line_black_adaptive,
        calibrate_lane_width
    )
    from utils.config_loader import load_config
    print("✅ Import thành công!")
except ImportError as e:
    print(f"❌ Lỗi Import: {e}")
    sys.exit(1)


# ============================================================
# BƯỚC 1: CALIBRATION
# ============================================================
def run_calibration(test_file='test_full_hd.jpg'):
    """
    Calibration - Đo 25cm = ? pixels
    Quan trọng: PHẢI chạy trước khi test!
    """
    print("\n" + "="*70)
    print("🔧 BƯỚC 1: CALIBRATION - Đo độ rộng lane")
    print("="*70)
    
    if not os.path.exists(test_file):
        print(f"❌ Không tìm thấy: {test_file}")
        print(f"💡 Hãy chụp ảnh test bằng capture.py trước!")
        return None
    
    frame = cv2.imread(test_file)
    if frame is None:
        print(f"❌ Không đọc được: {test_file}")
        return None
    
    print(f"📸 Ảnh gốc: {frame.shape[1]}x{frame.shape[0]}")
    print(f"🔄 Sẽ tự động resize về 640x480 để xử lý...")
    
    # Hàm calibrate_lane_width() sẽ tự động resize
    lane_width_px = calibrate_lane_width(frame, show_result=False)
    
    return lane_width_px


# ============================================================
# BƯỚC 2: TEST ẢNH TĨNH - SO SÁNH 2 PHƯƠNG PHÁP
# ============================================================
def run_tests(test_files, lane_config=None):
    """
    Test 2 phương pháp: Hough Transform vs Adaptive Threshold
    Kết quả lưu ra file ảnh debug_*.jpg
    """
    print("\n" + "="*70)
    print("🧪 BƯỚC 2: TEST ẢNH TĨNH - So sánh 2 phương pháp")
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
        
        print(f"\n📸 Đọc ảnh: {filename} ({frame.shape[1]}x{frame.shape[0]})")
        
        # ====================================================
        # METHOD 1: Hough Transform (Phương pháp chính)
        # ====================================================
        try:
            error_hough, x_line_hough, center_x, debug_hough = detect_line(frame, config=lane_config)
            
            if error_hough > 20:
                action_hough = "Rẽ PHẢI  (->)"
            elif error_hough < -20:
                action_hough = "Rẽ TRÁI  (<-)"
            else:
                action_hough = "Đi THẲNG (^)"
            
            print(f"{filename:<25} | {'Hough Transform':<20} | {error_hough:<8} | {action_hough}")
            
            # Lưu ảnh debug
            out_file = f"debug_hough_{filename}"
            cv2.imwrite(out_file, debug_hough)
            print(f"  💾 Đã lưu: {out_file}")
            
        except Exception as e:
            print(f"{filename:<25} | ❌ Hough Error: {e}")
        
        # ====================================================
        # METHOD 2: Adaptive Threshold
        # ====================================================
        try:
            error_adaptive, x_line_adaptive, _, debug_adaptive = detect_line_black_adaptive(frame)
            
            if error_adaptive > 20:
                action_adaptive = "Rẽ PHẢI  (->)"
            elif error_adaptive < -20:
                action_adaptive = "Rẽ TRÁI  (<-)"
            else:
                action_adaptive = "Đi THẲNG (^)"
            
            print(f"{'':<25} | {'Adaptive Threshold':<20} | {error_adaptive:<8} | {action_adaptive}")
            
            # Lưu ảnh debug
            out_file = f"debug_adaptive_{filename}"
            cv2.imwrite(out_file, debug_adaptive)
            print(f"  💾 Đã lưu: {out_file}")
            
            # So sánh 2 phương pháp
            diff = abs(error_hough - error_adaptive)
            if diff > 50:
                print(f"  ⚠️  Chênh lệch lớn: {diff}px")
            else:
                print(f"  ✅ Chênh lệch chấp nhận được: {diff}px")
                
        except Exception as e:
            print(f"{'':<25} | ❌ Adaptive Error: {e}")
        
        print("-" * 70)


# ============================================================
# BƯỚC 3: TEST REAL-TIME (Không dùng cv2.imshow)
# ============================================================
def test_realtime_camera(num_frames=50):
    """
    Test real-time với Picamera2
    Lưu mỗi 10 frames một lần để kiểm tra
    KHÔNG dùng cv2.imshow() (không có màn hình)
    """
    try:
        from picamera2 import Picamera2
        import time
        
        print("\n" + "="*70)
        print("🎥 BƯỚC 3: TEST REAL-TIME VỚI CAMERA")
        print("="*70)
        print(f"Sẽ chạy {num_frames} frames và lưu mỗi 10 frames")
        print("Không hiển thị (cv2.imshow) vì không có màn hình ngoài")
        
        picam2 = Picamera2()
        
        # Cấu hình camera 1640x1232 (Full FOV)
        config = picam2.create_preview_configuration(
            main={"size": (1640, 1232), "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()
        
        print("⏳ Warm-up camera 2 giây...")
        time.sleep(2)
        
        method = 'hough'  # Có thể đổi thành 'adaptive'
        frame_count = 0
        
        print(f"\n🚀 Bắt đầu xử lý {num_frames} frames...")
        
        while frame_count < num_frames:
            # Capture frame (RGB)
            frame_rgb = picam2.capture_array()
            
            # Chuyển sang BGR (OpenCV format)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Detect lane (hàm sẽ tự động resize về 640x480)
            if method == 'hough':
                error, x_line, center_x, debug_frame = detect_line(frame_bgr)
            else:
                error, x_line, center_x, debug_frame = detect_line_black_adaptive(frame_bgr)
            
            # In progress
            print(f"\rFrame {frame_count+1}/{num_frames} | Error: {error:+4d}px", end="", flush=True)
            
            # Lưu mỗi 10 frames
            if frame_count % 10 == 0:
                filename = f"realtime_frame_{frame_count:03d}.jpg"
                cv2.imwrite(filename, debug_frame)
            
            frame_count += 1
            
            # Delay nhỏ để không chiếm CPU
            time.sleep(0.05)
        
        print("\n")
        picam2.stop()
        picam2.close()
        
        print(f"✅ Đã xử lý {frame_count} frames")
        print(f"📸 Đã lưu {frame_count // 10} ảnh debug: realtime_frame_*.jpg")
        
    except ImportError:
        print("❌ Không tìm thấy Picamera2. Bỏ qua test real-time.")
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()


# ============================================================
# DIAGNOSTIC: Kiểm tra nhanh 1 ảnh
# ============================================================
def quick_diagnostic(image_path):
    """
    Chẩn đoán nhanh 1 ảnh để tìm lỗi
    """
    print("\n" + "="*70)
    print("🔍 CHẨN ĐOÁN NHANH")
    print("="*70)
    
    if not os.path.exists(image_path):
        print(f"❌ Không tìm thấy: {image_path}")
        return
    
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Không đọc được: {image_path}")
        return
    
    print(f"📸 Ảnh: {image_path}")
    print(f"   Kích thước gốc: {frame.shape[1]}x{frame.shape[0]}")
    
    # Resize về 640x480
    if frame.shape[1] != 640:
        frame_resized = cv2.resize(frame, (640, 480))
        print(f"   Đã resize về: 640x480")
    else:
        frame_resized = frame
    
    # Chuyển grayscale và đảo màu
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    gray_inv = cv2.bitwise_not(gray)
    
    # Lưu các bước xử lý
    cv2.imwrite("diag_1_original.jpg", frame_resized)
    cv2.imwrite("diag_2_gray.jpg", gray)
    cv2.imwrite("diag_3_inverted.jpg", gray_inv)
    
    # Canny
    edges = cv2.Canny(gray_inv, 40, 120)
    cv2.imwrite("diag_4_edges.jpg", edges)
    
    # Hough Lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 20, minLineLength=30, maxLineGap=20)
    
    if lines is not None:
        print(f"   ✅ Tìm thấy {len(lines)} đường thẳng")
        
        # Vẽ tất cả lines
        lines_img = frame_resized.copy()
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lines_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.imwrite("diag_5_lines.jpg", lines_img)
        print(f"   📸 Đã lưu: diag_*.jpg (5 files)")
    else:
        print(f"   ❌ KHÔNG tìm thấy đường thẳng nào!")
        print(f"   💡 Nguyên nhân có thể:")
        print(f"      - Vạch đen quá mờ")
        print(f"      - Ánh sáng quá yếu/chói")
        print(f"      - Vạch không nằm trong ROI")
        print(f"      - Tham số Canny/Hough quá cao")
    
    print("="*70)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚗 CHƯƠNG TRÌNH TEST LANE DETECTION (No Display Version)")
    print("   Tối ưu cho: Vạch ĐEN trên nền TRẮNG (bìa trắng)")
    print("   Resolution: 1640x1232 → Auto resize 640x480")
    print("   Không dùng cv2.imshow() (không có màn hình ngoài)")
    print("="*70)
    
    # Load config
    try:
        config_full = load_config('config/hardware_config.yaml')
        lane_config = config_full.get('ai', {}).get('lane_detection', {})
        print("✅ Đã tải config")
    except:
        print("⚠️  Dùng config mặc định")
        lane_config = None
    
    # Test files
    test_files = [
        'test_full_hd.jpg',
        'road_curve_left.jpg', 
        'road_curve_right.jpg'
    ]
    
    # Menu
    print("\n📋 Chọn chế độ test:")
    print("  1. CALIBRATION - Đo lane width (⚠️ BẮT BUỘC chạy trước!)")
    print("  2. TEST ẢNH TĨNH - So sánh Hough vs Adaptive")
    print("  3. TEST REAL-TIME - Camera (Lưu ảnh, không hiển thị)")
    print("  4. TẤT CẢ (Khuyến nghị)")
    print("  5. CHẨN ĐOÁN NHANH - Kiểm tra 1 ảnh chi tiết")
    
    choice = input("\n👉 Nhập lựa chọn (1-5): ").strip()
    
    if choice == '1':
        # CALIBRATION
        run_calibration('test_full_hd.jpg')
    
    elif choice == '2':
        # TEST ẢNH TĨNH
        run_tests(test_files, lane_config)
        print("\n✅ Hoàn tất! Kiểm tra file debug_*.jpg")
    
    elif choice == '3':
        # TEST REAL-TIME
        num_frames = int(input("Số frames cần test (mặc định 50): ") or "50")
        test_realtime_camera(num_frames)
    
    elif choice == '4':
        # TẤT CẢ
        print("\n🚀 Chạy quy trình đầy đủ...\n")
        
        # Step 1: Calibration
        lane_width_px = run_calibration('test_full_hd.jpg')
        
        if lane_width_px:
            input("\n✅ Calibration xong. Nhấn Enter tiếp tục...")
        else:
            print("⚠️  Calibration thất bại, nhưng vẫn tiếp tục test...")
        
        # Step 2: Test ảnh tĩnh
        run_tests(test_files, lane_config)
        
        print("\n✅ Test ảnh tĩnh xong!")
        
        # Step 3: Test real-time
        do_realtime = input("\nTest real-time với camera? (y/n): ")
        if do_realtime.lower() == 'y':
            num_frames = int(input("Số frames (mặc định 50): ") or "50")
            test_realtime_camera(num_frames)
    
    elif choice == '5':
        # CHẨN ĐOÁN
        image = input("Nhập tên file ảnh (mặc định test_full_hd.jpg): ").strip()
        if not image:
            image = 'test_full_hd.jpg'
        quick_diagnostic(image)
    
    else:
        print("❌ Lựa chọn không hợp lệ!")
    
    print("\n" + "="*70)
    print("✅ KẾT THÚC")
    print("="*70)
