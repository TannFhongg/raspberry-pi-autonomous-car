"""
Motor Test Script for L298N Driver
Tests all motor functions safely
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from drivers.motor.l298n_driver import L298NDriver
from utils.config_loader import load_config
from utils.logger import setup_logger

# Setup logger
logger = setup_logger('test_motor', level='INFO')


def test_motor_basic():
    """Basic motor test"""
    logger.info("=" * 60)
    logger.info("L298N Motor Driver Test - Basic")
    logger.info("=" * 60)
    
    try:
        # Load config
        config = load_config('config/hardware_config.yaml')
        logger.info("✅ Configuration loaded")
        
        # Initialize driver
        driver = L298NDriver(config, use_pigpio=False)
        logger.info("✅ Driver initialized")
        
        print("\n🚗 Starting motor test sequence...")
        print("⚠️  Make sure robot wheels are off the ground!\n")
        input("Press ENTER to start test...")
        
        # Test 1: Forward
        print("\n[Test 1/6] Moving FORWARD at speed 100...")
        driver.forward(100)
        time.sleep(2)
        driver.stop()
        time.sleep(1)
        
        # Test 2: Backward
        print("[Test 2/6] Moving BACKWARD at speed 100...")
        driver.backward(100)
        time.sleep(2)
        driver.stop()
        time.sleep(1)
        
        # Test 3: Turn Left
        print("[Test 3/6] Turning LEFT...")
        driver.turn_left(100)
        time.sleep(2)
        driver.stop()
        time.sleep(1)
        
        # Test 4: Turn Right
        print("[Test 4/6] Turning RIGHT...")
        driver.turn_right(100)
        time.sleep(2)
        driver.stop()
        time.sleep(1)
        
        # Test 5: Speed test
        print("[Test 5/6] Speed test (50, 150, 200)...")
        for speed in [50, 150, 200]:
            print(f"  Speed: {speed}")
            driver.forward(speed)
            time.sleep(1.5)
        driver.stop()
        time.sleep(1)
        
        # Test 6: Individual motors
        print("[Test 6/6] Testing individual motors...")
        print("  Left motor only...")
        driver.set_motors(150, 0)
        time.sleep(2)
        driver.stop()
        time.sleep(0.5)
        
        print("  Right motor only...")
        driver.set_motors(0, 150)
        time.sleep(2)
        driver.stop()
        
        print("\n✅ All tests completed successfully!")
        
    except FileNotFoundError as e:
        logger.error(f"❌ Config file not found: {e}")
        logger.error("Make sure config/hardware_config.yaml exists")
        return False
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        try:
            driver.cleanup()
            logger.info("✅ Cleanup completed")
        except:
            pass
    
    return True


def test_motor_interactive():
    """Interactive motor control"""
    logger.info("=" * 60)
    logger.info("L298N Motor Driver Test - Interactive Mode")
    logger.info("=" * 60)
    
    try:
        # Load config
        config = load_config('config/hardware_config.yaml')
        driver = L298NDriver(config, use_pigpio=False)
        logger.info("✅ Driver initialized")
        
        print("\n🎮 Interactive Motor Control")
        print("Commands:")
        print("  w - Forward")
        print("  s - Backward")
        print("  a - Turn Left")
        print("  d - Turn Right")
        print("  x - Stop")
        print("  + - Increase speed")
        print("  - - Decrease speed")
        print("  q - Quit")
        print()
        
        speed = 150
        running = True
        
        # Disable input buffering
        import sys
        import tty
        import termios
        
        old_settings = termios.tcgetattr(sys.stdin)
        
        try:
            tty.setcbreak(sys.stdin.fileno())
            
            print(f"Current speed: {speed}")
            print("Ready! (press keys to control)")
            
            while running:
                # Read single character
                char = sys.stdin.read(1).lower()
                
                if char == 'w':
                    print(f"\r↑ Forward ({speed})    ", end='', flush=True)
                    driver.forward(speed)
                    
                elif char == 's':
                    print(f"\r↓ Backward ({speed})   ", end='', flush=True)
                    driver.backward(speed)
                    
                elif char == 'a':
                    print(f"\r← Left ({speed})       ", end='', flush=True)
                    driver.turn_left(speed)
                    
                elif char == 'd':
                    print(f"\r→ Right ({speed})      ", end='', flush=True)
                    driver.turn_right(speed)
                    
                elif char == 'x':
                    print(f"\r■ Stop              ", end='', flush=True)
                    driver.stop()
                    
                elif char == '+' or char == '=':
                    speed = min(255, speed + 10)
                    print(f"\r+ Speed: {speed}       ", end='', flush=True)
                    
                elif char == '-' or char == '_':
                    speed = max(0, speed - 10)
                    print(f"\r- Speed: {speed}       ", end='', flush=True)
                    
                elif char == 'q':
                    print("\r\nQuitting...         ")
                    running = False
                    
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            driver.stop()
            driver.cleanup()
            
        print("✅ Interactive test completed")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_motor_check_wiring():
    """Check motor wiring and GPIO pins"""
    logger.info("=" * 60)
    logger.info("Motor Wiring Check")
    logger.info("=" * 60)
    
    try:
        config = load_config('config/hardware_config.yaml')
        
        print("\n📋 Configuration Check:")
        print(f"\nLeft Motor:")
        print(f"  Enable Pin (ENA): GPIO {config['motor_driver']['left_motor']['enable_pin']}")
        print(f"  Input 1 (IN1):    GPIO {config['motor_driver']['left_motor']['input1_pin']}")
        print(f"  Input 2 (IN2):    GPIO {config['motor_driver']['left_motor']['input2_pin']}")
        print(f"  Reverse:          {config['motor_driver']['left_motor']['reverse']}")
        
        print(f"\nRight Motor:")
        print(f"  Enable Pin (ENB): GPIO {config['motor_driver']['right_motor']['enable_pin']}")
        print(f"  Input 1 (IN3):    GPIO {config['motor_driver']['right_motor']['input1_pin']}")
        print(f"  Input 2 (IN4):    GPIO {config['motor_driver']['right_motor']['input2_pin']}")
        print(f"  Reverse:          {config['motor_driver']['right_motor']['reverse']}")
        
        print(f"\nPWM Frequency: {config['pwm']['frequency']} Hz")
        
        print("\n✅ Configuration loaded successfully")
        print("\nℹ️  Verify these pins match your L298N wiring!")
        
    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        return False
    
    return True


def main():
    """Main test menu"""
    print("=" * 60)
    print("🚗 L298N Motor Driver Test Suite")
    print("=" * 60)
    print("\nSelect test mode:")
    print("  1 - Basic Test (automated sequence)")
    print("  2 - Interactive Control (keyboard)")
    print("  3 - Check Wiring Configuration")
    print("  q - Quit")
    print()
    
    choice = input("Enter choice: ").strip().lower()
    
    if choice == '1':
        test_motor_basic()
    elif choice == '2':
        test_motor_interactive()
    elif choice == '3':
        test_motor_check_wiring()
    elif choice == 'q':
        print("Goodbye!")
    else:
        print("Invalid choice!")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        print("Cleaning up...")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()