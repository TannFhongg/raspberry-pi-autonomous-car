"""
Test Arduino Connection và Communication
Chạy: python review_tool/test_arduino.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import serial
import serial.tools.list_ports
import time
from utils.config_loader import load_config


def list_serial_ports():
    """Liệt kê tất cả serial ports"""
    print("=" * 60)
    print("SERIAL PORTS AVAILABLE")
    print("=" * 60)
    
    ports = serial.tools.list_ports.comports()
    
    if not ports:
        print("❌ Không tìm thấy serial port nào!")
        return []
    
    for i, port in enumerate(ports):
        print(f"\n{i+1}. {port.device}")
        print(f"   Description: {port.description}")
        print(f"   Hardware ID: {port.hwid}")
        
        if 'Arduino' in port.description or 'USB' in port.description:
            print(f"   ✅ Có thể là Arduino")
    
    return [port.device for port in ports]


def test_port(port, baudrate=115200, timeout=2):
    """Test kết nối với port cụ thể"""
    print(f"\n" + "=" * 60)
    print(f"Testing port: {port}")
    print("=" * 60)
    
    try:
        # Mở serial port
        print(f"1. Mở port {port} @ {baudrate} baud...")
        ser = serial.Serial(port, baudrate, timeout=timeout)
        print(f"   ✅ Port opened")
        
        # Đợi Arduino reset
        print(f"2. Đợi Arduino reset (2 giây)...")
        time.sleep(2)
        
        # Đọc startup message
        if ser.in_waiting > 0:
            startup = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
            print(f"   📥 Startup: {startup.strip()}")
        
        # Gửi PING command
        print(f"3. Gửi PING command...")
        command = '{"cmd":"PING"}\n'
        ser.write(command.encode('utf-8'))
        ser.flush()
        
        # Đợi PONG response
        print(f"4. Đợi PONG response...")
        start_time = time.time()
        response = ""
        
        while time.time() - start_time < timeout:
            if ser.in_waiting > 0:
                response += ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                if 'PONG' in response or 'ok' in response:
                    print(f"   ✅ PONG received!")
                    print(f"   📥 Response: {response.strip()}")
                    ser.close()
                    return True
            time.sleep(0.1)
        
        print(f"   ❌ Không nhận được PONG")
        if response:
            print(f"   📥 Nhận được: {response.strip()}")
        
        ser.close()
        return False
        
    except serial.SerialException as e:
        print(f"   ❌ Serial error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Lỗi: {e}")
        return False


def main():
    print("\n" + "🔧 " * 20)
    print("ARDUINO CONNECTION TEST")
    print("🔧 " * 20)
    
    # List all ports
    ports = list_serial_ports()
    
    if not ports:
        print("\n❌ Không tìm thấy serial port!")
        print("\n💡 Kiểm tra:")
        print("  1. Arduino có kết nối USB không?")
        print("  2. Cable USB có hoạt động không?")
        print("  3. Driver đã cài chưa?")
        return
    
    # Load config để lấy port mặc định
    try:
        config = load_config('config/hardware_config.yaml')
        default_port = config.get('arduino', {}).get('port', '/dev/ttyACM0')
        print(f"\n📋 Port mặc định từ config: {default_port}")
    except:
        default_port = None
    
    # Test các ports
    print("\n" + "=" * 60)
    print("TESTING PORTS")
    print("=" * 60)
    
    working_ports = []
    
    for port in ports:
        if test_port(port):
            working_ports.append(port)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if working_ports:
        print(f"\n✅ Tìm thấy {len(working_ports)} Arduino:")
        for port in working_ports:
            print(f"   • {port}")
        
        print(f"\n💡 Cập nhật hardware_config.yaml:")
        print(f"   arduino:")
        print(f"     port: '{working_ports[0]}'")
        print(f"     baudrate: 115200")
    else:
        print(f"\n❌ Không tìm thấy Arduino hoạt động!")
        print(f"\n💡 Troubleshooting:")
        print(f"  1. Kiểm tra Arduino đã cắm USB chưa")
        print(f"  2. Thử cable USB khác")
        print(f"  3. Kiểm tra Arduino IDE có kết nối được không")
        print(f"  4. Re-upload firmware lên Arduino")


if __name__ == '__main__':
    main()
