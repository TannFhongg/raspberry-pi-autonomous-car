"""
Test Motor Control - Kiểm tra motor driver
Chạy: python review_tool/test_motor.py
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from drivers.motor.arduino_driver import ArduinoDriver
from utils.config_loader import load_config


def test_motor_basic():
    """Test cơ bản motor"""
    print("=" * 60)
    print("MOTOR TEST - BASIC")
    print("=" * 60)
    
    try:
        # Load config
        config = load_config('config/hardware_config.yaml')
        print("✅ Config loaded")
        
        # Check control mode
        control_mode = config.get('control_mode', 'arduino')
        print(f"Control mode: {control_mode}")
        
        if control_mode != 'arduino':
            print("❌ Test này chỉ dành cho Arduino mode")
            print("💡 Đổi control_mode = 'arduino' trong hardware_config.yaml")
            return False
        
        # Initialize Arduino driver
        arduino_config = config.get('arduino', {})
        print(f"\nKết nối Arduino: {arduino_config.get('port', '/dev/ttyACM0')}")
        
        driver = ArduinoDriver(
            port=arduino_config.get('port', '/dev/ttyACM0'),
            baudrate=arduino_config.get('baudrate', 115200)
        )
        
        if not driver.connected:
            print("❌ Không kết nối được Arduino!")
            print("\n💡 Kiểm tra:")
            print("  1. Arduino có kết nối USB không?")
            print("  2. Port đúng không? (/dev/ttyACM0, /dev/ttyUSB0...)")
            print("  3. Firmware đã upload chưa?")
            print("  4. User có quyền truy cập serial không?")
            return False
        
        print("✅ Arduino connected")
        
        print("\n⚠️  CẢNH BÁO: Đảm bảo robot có thể di chuyển an toàn!")
        input("Nhấn ENTER để bắt đầu test motor...")
        
        # Test sequence
        print("\n" + "=" * 60)
        print("TEST SEQUENCE")
        print("=" * 60)
        
        # Test 1: Forward
        print("\n[1/6] FORWARD (speed 100)...")
        driver.forward(100)
        time.sleep(2)
        driver.stop()
        time.sleep(1)
        
        # Test 2: Backward
        print("[2/6] BACKWARD (speed 100)...")
        driver.backward(100)
        time.sleep(2)
        driver.stop()
        time.sleep(1)
        
        # Test 3: Turn Left
        print("[3/6] TURN LEFT...")
        driver.turn_left(100)
        time.sleep(2)
        driver.stop()
        time.sleep(1)
        
        # Test 4: Turn Right
        print("[4/6] TURN RIGHT...")
        driver.turn_right(100)
        time.sleep(2)
        driver.stop()
        time.sleep(1)
        
        # Test 5: Speed variations
        print("[5/6] SPEED TEST (50, 150, 200)...")
        for speed in [50, 150, 200]:
            print(f"  Speed: {speed}")
            driver.forward(speed)
            time.sleep(1.5)
        driver.stop()
        time.sleep(1)
        
        # Test 6: Individual motors
        print("[6/6] INDIVIDUAL MOTORS...")
        print("  Left motor FORWARD only...")
        driver.set_motors(150, 0)
        time.sleep(2)
        driver.stop()
        time.sleep(0.5)

        print("  Left motor BACKWARD only...")
        driver.set_motors(-150, 0)
        time.sleep(2)
        driver.stop()
        time.sleep(0.5)

        print("  Right motor FORWARD only...")
        driver.set_motors(0, 150)
        time.sleep(2)
        driver.stop()
        time.sleep(0.5)

        print("  Right motor BACKWARD only...")
        driver.set_motors(0, -150)
        time.sleep(2)
        driver.stop()
        
        print("\n✅ Tất cả test đã hoàn thành!")
        
        # Cleanup
        driver.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Lỗi test motor: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_motor_speeds():
    """Test các mức tốc độ khác nhau"""
    print("=" * 60)
    print("MOTOR TEST - SPEED RANGE")
    print("=" * 60)
    
    try:
        config = load_config('config/hardware_config.yaml')
        arduino_config = config.get('arduino', {})
        
        driver = ArduinoDriver(
            port=arduino_config.get('port', '/dev/ttyACM0'),
            baudrate=arduino_config.get('baudrate', 115200)
        )
        
        if not driver.connected:
            print("❌ Không kết nối được Arduino!")
            return False
        
        print("✅ Arduino connected")
        print("\n⚠️  Robot sẽ di chuyển với các tốc độ khác nhau")
        input("Nhấn ENTER để bắt đầu...")
        
        speeds = [50, 100, 150, 200, 255]
        
        for speed in speeds:
            print(f"\nForward - Speed: {speed}")
            driver.forward(speed)
            time.sleep(2)
            driver.stop()
            time.sleep(1)
        
        print("\n✅ Speed test hoàn thành!")
        
        driver.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return False


def main():
    print("\n" + "🔧 " * 20)
    print("MOTOR DRIVER TEST SUITE")
    print("🔧 " * 20)
    
    print("\nChọn test mode:")
    print("  1 - Basic Test (tất cả chức năng)")
    print("  2 - Speed Range Test (test các mức tốc độ)")
    print("  q - Quit")
    print()
    
    choice = input("Nhập lựa chọn: ").strip().lower()
    
    if choice == '1':
        test_motor_basic()
    elif choice == '2':
        test_motor_speeds()
    elif choice == 'q':
        print("Goodbye!")
    else:
        print("Lựa chọn không hợp lệ!")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test bị ngắt bởi user")
        print("Dọn dẹp...")
