#!/usr/bin/env python3
"""
Test Arduino Connection
Quick script to diagnose Arduino connectivity issues
"""

import serial
import serial.tools.list_ports
import time
import sys

def list_serial_ports():
    """List all available serial ports"""
    print("=" * 60)
    print("Available Serial Ports:")
    print("=" * 60)
    
    ports = serial.tools.list_ports.comports()
    
    if not ports:
        print("❌ No serial ports found!")
        return []
    
    for i, port in enumerate(ports):
        print(f"\n{i+1}. {port.device}")
        print(f"   Description: {port.description}")
        print(f"   Hardware ID: {port.hwid}")
        
        # Check if it's Arduino
        if 'Arduino' in port.description or 'USB' in port.description:
            print(f"   ✅ Likely Arduino device")
    
    return [port.device for port in ports]


def test_port(port, baudrate=115200, timeout=2):
    """Test connection to a specific port"""
    print(f"\n" + "=" * 60)
    print(f"Testing port: {port}")
    print("=" * 60)
    
    try:
        # Try to open serial port
        print(f"1. Opening port {port} at {baudrate} baud...")
        ser = serial.Serial(port, baudrate, timeout=timeout)
        print(f"   ✅ Port opened successfully")
        
        # Wait for Arduino to reset
        print(f"2. Waiting for Arduino to reset (2 seconds)...")
        time.sleep(2)
        
        # Clear any startup messages
        if ser.in_waiting > 0:
            startup = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
            print(f"   📥 Startup message: {startup.strip()}")
        
        # Send PING command
        print(f"3. Sending PING command...")
        ser.write(b'PING\n')
        ser.flush()
        
        # Wait for response
        print(f"4. Waiting for PONG response...")
        start_time = time.time()
        response = ""
        
        while time.time() - start_time < timeout:
            if ser.in_waiting > 0:
                response += ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                if 'PONG' in response:
                    print(f"   ✅ PONG received!")
                    print(f"   📥 Full response: {response.strip()}")
                    ser.close()
                    return True
            time.sleep(0.1)
        
        print(f"   ❌ No PONG response received")
        if response:
            print(f"   📥 Got instead: {response.strip()}")
        
        ser.close()
        return False
        
    except serial.SerialException as e:
        print(f"   ❌ Serial error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_permissions(port):
    """Check file permissions on serial port"""
    import os
    import pwd
    import grp
    
    print(f"\n" + "=" * 60)
    print(f"Permission Check: {port}")
    print("=" * 60)
    
    try:
        # Get file stats
        stat_info = os.stat(port)
        
        # Get owner and group
        owner = pwd.getpwuid(stat_info.st_uid).pw_name
        group = grp.getgrgid(stat_info.st_gid).gr_name
        
        # Get permissions
        mode = oct(stat_info.st_mode)[-3:]
        
        print(f"Owner: {owner}")
        print(f"Group: {group}")
        print(f"Permissions: {mode}")
        
        # Check if current user is in the group
        import subprocess
        result = subprocess.run(['groups'], capture_output=True, text=True)
        user_groups = result.stdout.strip().split()
        
        print(f"\nYour groups: {', '.join(user_groups)}")
        
        if group in user_groups:
            print(f"✅ You are in the '{group}' group")
        else:
            print(f"❌ You are NOT in the '{group}' group")
            print(f"\nTo fix, run:")
            print(f"  sudo usermod -a -G {group} $USER")
            print(f"  newgrp {group}  # Or logout and login")
        
        return group in user_groups
        
    except Exception as e:
        print(f"❌ Error checking permissions: {e}")
        return False


def main():
    """Main test function"""
    print("\n" + "🔧 " * 20)
    print("Arduino Connection Diagnostic Tool")
    print("🔧 " * 20)
    
    # Step 1: List all ports
    ports = list_serial_ports()
    
    if not ports:
        print("\n❌ No serial ports detected!")
        print("\nPossible issues:")
        print("  1. Arduino is not connected")
        print("  2. USB cable is faulty (try another cable)")
        print("  3. Arduino drivers not installed")
        sys.exit(1)
    
    # Step 2: Check permissions on common ports
    common_ports = ['/dev/ttyACM0', '/dev/ttyUSB0']
    for port in common_ports:
        if port in ports:
            check_permissions(port)
    
    # Step 3: Test each port
    print("\n" + "=" * 60)
    print("Testing each port...")
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
        print(f"\n✅ Working Arduino ports found:")
        for port in working_ports:
            print(f"   • {port}")
        
        print(f"\n💡 Update your hardware_config.yaml:")
        print(f"   arduino:")
        print(f"     port: '{working_ports[0]}'")
        print(f"     baudrate: 115200")
        
    else:
        print(f"\n❌ No working Arduino connections found!")
        print(f"\nTroubleshooting steps:")
        print(f"  1. Check Arduino is plugged in")
        print(f"  2. Try a different USB cable")
        print(f"  3. Check Arduino IDE can connect to the board")
        print(f"  4. Re-upload the firmware to Arduino")
        print(f"  5. Check for permission issues (see above)")
        print(f"  6. Try different USB ports on Raspberry Pi")


if __name__ == '__main__':
    main()