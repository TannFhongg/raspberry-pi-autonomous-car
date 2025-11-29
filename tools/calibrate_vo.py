#!/usr/bin/env python3
"""
Visual Odometry Calibration Tool (Headless / SSH Mode)
Chạy trực tiếp trên Terminal, không cần màn hình.
Tích hợp CameraManager cho Raspberry Pi 5.
"""

import sys
import time
import select
import tty
import termios
import os
import cv2
import logging

# Tắt bớt log rác để giao diện sạch đẹp
logging.basicConfig(level=logging.INFO)
logging.getLogger('perception.camera_manager').setLevel(logging.WARNING)
logging.getLogger('perception.visual_odometry').setLevel(logging.WARNING)
logging.getLogger('picamera2').setLevel(logging.WARNING)

# Thêm đường dẫn project để import module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from perception.visual_odometry import VisualOdometry
from perception.camera_manager import CameraManager
from utils.config_loader import load_config

def is_data():
    """Kiểm tra có phím nào được nhấn không (Non-blocking)"""
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

def main():
    print("\n" + "="*60)
    print("📏 VO CALIBRATION TOOL (SSH MODE)")
    print("="*60 + "\n")
    
    # 1. Khởi tạo CameraManager
    try:
        print("📷 Đang khởi động Camera...")
        config_path = os.path.join(parent_dir, 'config/hardware_config.yaml')
        config = load_config(config_path)
        
        camera = CameraManager(config)
        if not camera.start():
            print("❌ Lỗi: Không thể khởi động CameraManager!")
            return
        print("✅ Camera OK.")
    except Exception as e:
        print(f"❌ Lỗi khởi tạo: {e}")
        return
    
    # 2. Khởi tạo VO
    # Load scale cũ nếu có
    old_scale = config.get('visual_odometry', {}).get('scale_factor', 1.0)
    vo = VisualOdometry(scale_factor=old_scale)
    
    print(f"ℹ️  Scale hiện tại: {old_scale}")
    print("\n🎮 HƯỚNG DẪN:")
    print("  1. Đặt robot tại vạch xuất phát.")
    print("  2. Nhấn 's' để BẮT ĐẦU đo (START).")
    print("  3. Đẩy robot đi thẳng một đoạn (ví dụ 50cm).")
    print("  4. Nhấn 'e' để KẾT THÚC đo (END).")
    print("  5. Nhập khoảng cách thực tế.")
    print("  6. Nhấn 'q' để thoát.\n")
    
    calibration_started = False
    
    # Cấu hình terminal để đọc phím không cần Enter
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        
        print("⏳ Đang chạy... (Nhấn 's' để bắt đầu)\n")
        
        while True:
            # Lấy frame từ CameraManager
            frame = camera.capture_frame()
            
            if frame is None:
                time.sleep(0.01)
                continue
            
            # Chuyển đổi màu sắc (Picamera2 RGB -> OpenCV BGR)
            # Quan trọng để thuật toán chuyển xám hoạt động đúng
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Tính toán VO
            vo.process_frame(frame_bgr)
            status = vo.get_status()
            
            # Hiển thị thông tin (Ghi đè dòng cũ)
            y_px = status['position_y_px']
            quality = status['tracking_quality']
            feats = status['num_features']
            
            mode_str = "🔴 REC" if calibration_started else "⚪ IDLE"
            
            # In ra màn hình console
            sys.stdout.write(f"\r[{mode_str}] Y: {y_px:8.1f} px | Feat: {feats:3} | Qual: {quality:.2f}   ")
            sys.stdout.flush()
            
            # Xử lý phím bấm
            if is_data():
                key = sys.stdin.read(1).lower()
                
                if key == 'q':
                    print("\n\n👋 Thoát.")
                    break
                
                elif key == 's': # START
                    if not calibration_started:
                        vo.reset()
                        calibration_started = True
                        print("\n\n🟢 BẮT ĐẦU! Hãy di chuyển robot...")
                
                elif key == 'e': # END
                    if calibration_started:
                        measured_pixels = abs(status['position_y_px'])
                        print(f"\n\n🛑 DỪNG LẠI.")
                        print(f"📏 Máy đo được: {measured_pixels:.1f} pixels")
                        
                        # Khôi phục terminal để nhập liệu
                        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                        
                        try:
                            if measured_pixels < 10:
                                print("⚠️ Chuyển động quá nhỏ (<10px), không thể tính toán.")
                            else:
                                real_dist_input = input("✍️  Nhập khoảng cách thực tế (cm): ")
                                real_dist = float(real_dist_input)
                                
                                scale = real_dist / measured_pixels
                                print(f"\n✅ KẾT QUẢ:")
                                print(f"   Scale Factor: {scale:.5f}")
                                print("-" * 40)
                                print(f"👉 Cập nhật file 'config/hardware_config.yaml':")
                                print(f"   visual_odometry:")
                                print(f"     scale_factor: {scale:.5f}")
                                print("-" * 40)
                        except ValueError:
                            print("❌ Lỗi nhập liệu. Vui lòng nhập số.")
                            
                        # Thiết lập lại chế độ phím để đo tiếp
                        tty.setcbreak(sys.stdin.fileno())
                        calibration_started = False
                        print("\nNhấn 's' để đo lại, 'q' để thoát.\n")

            time.sleep(0.05)
            
    except Exception as e:
        print(f"\n\n❌ Lỗi Runtime: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Dọn dẹp và khôi phục terminal
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        if 'camera' in locals() and camera:
            camera.stop()
        print("\nCamera released. Goodbye!")

if __name__ == "__main__":
    main()