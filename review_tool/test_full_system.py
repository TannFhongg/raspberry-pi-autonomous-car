"""
Test Full System Integration
Kiểm tra toàn bộ hệ thống: Camera → Lane Detection → PID → Motor
Chạy: python review_tool/test_full_system.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import cv2
from utils.config_loader import load_config
from perception.camera_manager import CameraManager
from perception.lane_detector import detect_line
from control.pid_controller import PIDController
from drivers.motor.arduino_driver import ArduinoDriver


def test_full_system():
    """Test toàn bộ hệ thống"""
    print("=" * 60)
    print("FULL SYSTEM INTEGRATION TEST")
    print("=" * 60)
    
    # Load config
    try:
        config = load_config('config/hardware_config.yaml')
        print("✅ Config loaded")
    except Exception as e:
        print(f"❌ Không load được config: {e}")
        return False
    
    # Initialize components
    print("\n" + "=" * 60)
    print("INITIALIZE COMPONENTS")
    print("=" * 60)
    
    # 1. Camera
    print("\n[1/3] Camera...")
    camera = CameraManager(config)
    if not camera.start():
        print("❌ Camera initialization failed")
        return False
    print("✅ Camera OK")
    
    # 2. Arduino Motor Driver
    print("\n[2/3] Arduino Motor Driver...")
    control_mode = config.get('control_mode', 'arduino')
    
    if control_mode != 'arduino':
        print("❌ Test này yêu cầu Arduino mode")
        camera.stop()
        return False
    
    arduino_config = config.get('arduino', {})
    driver = ArduinoDriver(
        port=arduino_config.get('port', '/dev/ttyACM0'),
        baudrate=arduino_config.get('baudrate', 115200)
    )
    
    if not driver.connected:
        print("❌ Arduino connection failed")
        camera.stop()
        return False
    print("✅ Arduino OK")
    
    # 3. PID Controller
    print("\n[3/3] PID Controller...")
    pid_cfg = config.get('lane_following', {}).get('pid', {})
    pid = PIDController(
        kp=pid_cfg.get('kp', 0.45),
        ki=pid_cfg.get('ki', 0.002),
        kd=pid_cfg.get('kd', 0.08),
        output_min=-255,
        output_max=255
    )
    print("✅ PID OK")
    
    print("\n✅ TẤT CẢ COMPONENTS ĐÃ SẴN SÀNG")
    
    # Integration test
    print("\n" + "=" * 60)
    print("INTEGRATION TEST")
    print("=" * 60)
    print("\n⚠️  Robot sẽ thử điều khiển dựa trên lane detection")
    print("⚠️  ĐẢM BẢO ROBOT AN TOÀN, CÓ THỂ DI CHUYỂN!")
    print()
    choice = input("Nhấn ENTER để bắt đầu (hoặc Ctrl+C để hủy): ")
    
    try:
        lane_config = config.get('ai', {}).get('lane_detection', {})
        base_speed = config.get('lane_following', {}).get('base_speed', 100)
        
        print(f"\nBase Speed: {base_speed}")
        print("Bắt đầu test loop (10 giây)...")
        print("Nhấn Ctrl+C để dừng")
        
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < 10.0:
            # 1. Capture frame
            frame = camera.capture_frame()
            if frame is None:
                print("⚠️  Frame capture failed")
                time.sleep(0.1)
                continue
            
            # 2. Lane detection
            error, x_line, center_x, debug_frame = detect_line(frame, lane_config)
            
            # 3. PID control
            dt = 0.03  # 30ms cycle
            correction = pid.compute(error, dt)
            
            # 4. Calculate motor speeds
            left_speed = int(base_speed - correction)
            right_speed = int(base_speed + correction)
            
            # Clamp speeds
            left_speed = max(-255, min(255, left_speed))
            right_speed = max(-255, min(255, right_speed))
            
            # 5. Send to motors
            driver.set_motors(left_speed, right_speed)
            
            # Print status
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"[{frame_count:3d}] Error: {error:+4d}px | "
                      f"Correction: {correction:+6.1f} | "
                      f"L: {left_speed:+4d} R: {right_speed:+4d}")
            
            time.sleep(0.03)
        
        # Stop motors
        driver.stop()
        
        # Statistics
        elapsed = time.time() - start_time
        fps = frame_count / elapsed
        
        print("\n" + "=" * 60)
        print("STATISTICS")
        print("=" * 60)
        print(f"Frames processed: {frame_count}")
        print(f"Time elapsed:     {elapsed:.2f}s")
        print(f"FPS:              {fps:.1f}")
        print(f"\n✅ Full system test PASSED")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test bị ngắt bởi user")
        driver.stop()
    
    finally:
        # Cleanup
        print("\nDọn dẹp...")
        driver.cleanup()
        camera.stop()
        print("✅ Cleanup hoàn tất")
    
    return True


def main():
    print("\n" + "🚀 " * 20)
    print("FULL SYSTEM INTEGRATION TEST")
    print("🚀 " * 20)
    
    print("\nTest này sẽ kiểm tra:")
    print("  ✓ Camera capture")
    print("  ✓ Lane detection")
    print("  ✓ PID controller")
    print("  ✓ Arduino motor control")
    print("  ✓ Full integration loop")
    
    print("\n⚠️  QUAN TRỌNG:")
    print("  • Robot phải có thể di chuyển an toàn")
    print("  • Đặt robot trên lane test hoặc nâng bánh xe lên")
    print("  • Sẵn sàng nhấn Ctrl+C để dừng khẩn cấp")
    
    print()
    choice = input("Tiếp tục? (y/n): ").strip().lower()
    
    if choice == 'y':
        test_full_system()
    else:
        print("Test bị hủy")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test bị ngắt")
