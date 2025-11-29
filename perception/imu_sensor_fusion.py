"""
IMU Sensor Fusion - FIXED VERSION
Fix cho lỗi [Errno 121] khi i2cdetect thấy 0x68

Vấn đề: Read quá nhanh sau khi wake up MPU-6050
Giải pháp: Thêm delay và retry logic
"""

import smbus2
import math
import time
import threading
import logging
from collections import deque

logger = logging.getLogger(__name__)


class IMUSensorFusion:
    """
    MPU-6050 với Complementary Filter - FIXED VERSION
    Thêm retry logic và proper initialization delays
    """
    
    # MPU-6050 Register Map
    PWR_MGMT_1 = 0x6B
    GYRO_CONFIG = 0x1B
    ACCEL_CONFIG = 0x1C
    TEMP_OUT_H = 0x41
    WHO_AM_I = 0x75
    
    ACCEL_XOUT_H = 0x3B
    ACCEL_YOUT_H = 0x3D
    ACCEL_ZOUT_H = 0x3F
    
    GYRO_XOUT_H = 0x43
    GYRO_YOUT_H = 0x45
    GYRO_ZOUT_H = 0x47
    
    def __init__(self, address=0x68, bus_num=1):
        self.bus = None
        self.address = address
        self.connected = False
        
        # Orientation (Fused)
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        
        # Complementary Filter coefficient
        self.ALPHA = 0.98
        
        # Timing
        self.prev_time = time.time()
        self.running = False
        self.thread = None
        
        # Calibration data
        self.gyro_x_offset = 0.0
        self.gyro_y_offset = 0.0
        self.gyro_z_offset = 0.0
        self.accel_x_offset = 0.0
        self.accel_y_offset = 0.0
        self.accel_z_offset = 1.0
        
        # Temperature
        self.temp_baseline = 0.0
        self.temp_drift_coeff = 0.01
        
        # Dynamic recalibration
        self.motion_history = deque(maxlen=50)
        self.STATIONARY_THRESHOLD = 0.5
        self.last_recalib_time = time.time()
        self.RECALIB_INTERVAL = 30.0
        
        # Connection monitoring
        self.last_successful_read = time.time()
        self.CONNECTION_TIMEOUT = 5.0
        
        # ===== THÊM: Retry counter =====
        self.read_error_count = 0
        self.MAX_READ_ERRORS = 5
        
        # Initialize hardware
        self._init_hardware(bus_num)
    
    def _init_hardware(self, bus_num):
        """Initialize MPU-6050 with proper delays"""
        try:
            self.bus = smbus2.SMBus(bus_num)
            logger.info("I2C bus opened successfully")
            
            # ===== CRITICAL FIX 1: Verify device presence =====
            logger.info("Verifying MPU-6050 presence...")
            max_retries = 3
            
            for attempt in range(max_retries):
                try:
                    who_am_i = self.bus.read_byte_data(self.address, self.WHO_AM_I)
                    
                    if who_am_i == 0x68:
                        logger.info(f"✅ MPU-6050 detected (WHO_AM_I = 0x{who_am_i:02X})")
                        break
                    else:
                        logger.warning(f"⚠️ Unexpected WHO_AM_I: 0x{who_am_i:02X}")
                        
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                    time.sleep(0.5)
                    
                    if attempt == max_retries - 1:
                        raise Exception("Failed to verify MPU-6050 after retries")
            
            # ===== CRITICAL FIX 2: Wake up with delay =====
            logger.info("Waking up MPU-6050...")
            self.bus.write_byte_data(self.address, self.PWR_MGMT_1, 0)
            time.sleep(0.3)  # ← INCREASED từ 0.1s → 0.3s
            
            # Verify wake up successful
            pwr_mgmt = self.bus.read_byte_data(self.address, self.PWR_MGMT_1)
            logger.info(f"Power Management register: 0x{pwr_mgmt:02X}")
            
            if pwr_mgmt != 0:
                logger.warning(f"⚠️ PWR_MGMT_1 not zero after wake: 0x{pwr_mgmt:02X}")
            
            # ===== CRITICAL FIX 3: Configure with delays =====
            logger.info("Configuring Gyro...")
            self.bus.write_byte_data(self.address, self.GYRO_CONFIG, 0)
            time.sleep(0.1)
            
            logger.info("Configuring Accelerometer...")
            self.bus.write_byte_data(self.address, self.ACCEL_CONFIG, 0)
            time.sleep(0.1)
            
            # ===== CRITICAL FIX 4: Test read temperature first =====
            logger.info("Testing sensor read...")
            temp = self._read_temperature_safe()
            logger.info(f"Initial temperature: {temp:.1f}°C")
            self.temp_baseline = temp
            
            self.connected = True
            logger.info("✅ MPU-6050 initialized successfully")
            
            # Calibrate
            self._calibrate()
            
        except Exception as e:
            logger.error(f"❌ MPU-6050 initialization failed: {e}")
            self.connected = False
            
            # Print troubleshooting hints
            logger.error("\n🔧 Troubleshooting:")
            logger.error("  1. Check wiring: VCC→Pin1, GND→Pin6, SDA→Pin3, SCL→Pin5")
            logger.error("  2. Verify 3.3V (not 5V!) is used for VCC")
            logger.error("  3. Run: sudo i2cdetect -y 1 (should see 68)")
            logger.error("  4. Check for loose connections")
    
    def _read_word_safe(self, reg, max_retries=3):
        """
        Read signed 16-bit word with retry logic
        FIX: Thêm retry để xử lý transient errors
        """
        for attempt in range(max_retries):
            try:
                h = self.bus.read_byte_data(self.address, reg)
                l = self.bus.read_byte_data(self.address, reg + 1)
                val = (h << 8) + l
                
                # Convert to signed
                if val >= 0x8000:
                    val = -((65535 - val) + 1)
                
                self.last_successful_read = time.time()
                self.read_error_count = 0  # Reset error counter on success
                return val
                
            except Exception as e:
                self.read_error_count += 1
                
                if attempt < max_retries - 1:
                    logger.warning(f"Read error at 0x{reg:02X}, retry {attempt + 1}/{max_retries}")
                    time.sleep(0.01)  # Small delay before retry
                else:
                    logger.error(f"❌ Read failed at register 0x{reg:02X} after {max_retries} attempts: {e}")
                    
                    # Check if too many consecutive errors
                    if self.read_error_count >= self.MAX_READ_ERRORS:
                        logger.error("⚠️ Too many read errors! MPU-6050 may have disconnected.")
                        self.connected = False
                    
                    self._check_connection()
                    return 0
    
    def _read_word(self, reg):
        """Wrapper cho compatibility"""
        return self._read_word_safe(reg)
    
    def _read_temperature_safe(self):
        """Read temperature with error handling"""
        try:
            temp_raw = self._read_word_safe(self.TEMP_OUT_H)
            temp_c = (temp_raw / 340.0) + 36.53
            return temp_c
        except:
            return self.temp_baseline
    
    def _read_temperature(self):
        """Wrapper cho compatibility"""
        return self._read_temperature_safe()
    
    def _check_connection(self):
        """Monitor connection health"""
        if time.time() - self.last_successful_read > self.CONNECTION_TIMEOUT:
            logger.error("❌ IMU connection lost!")
            self.connected = False
    
    def _calibrate(self, samples=500):
        """
        Calibrate gyro and accelerometer offsets
        FIX: Thêm error handling trong calibration
        """
        if not self.connected:
            return
        
        logger.info("🔧 Calibrating IMU... KEEP ROBOT STILL AND LEVEL!")
        
        sum_gx, sum_gy, sum_gz = 0, 0, 0
        sum_ax, sum_ay, sum_az = 0, 0, 0
        
        valid_samples = 0
        
        for i in range(samples):
            try:
                # Read raw values
                gx = self._read_word_safe(self.GYRO_XOUT_H)
                gy = self._read_word_safe(self.GYRO_YOUT_H)
                gz = self._read_word_safe(self.GYRO_ZOUT_H)
                
                ax = self._read_word_safe(self.ACCEL_XOUT_H)
                ay = self._read_word_safe(self.ACCEL_YOUT_H)
                az = self._read_word_safe(self.ACCEL_ZOUT_H)
                
                # Only count if read successful (non-zero)
                if gx != 0 or gy != 0 or gz != 0:
                    sum_gx += gx
                    sum_gy += gy
                    sum_gz += gz
                    
                    sum_ax += ax
                    sum_ay += ay
                    sum_az += az
                    
                    valid_samples += 1
                
                if i % 100 == 0:
                    logger.info(f"Calibration progress: {i}/{samples} ({valid_samples} valid)")
                
                time.sleep(0.01)
                
            except Exception as e:
                logger.warning(f"Calibration sample {i} failed: {e}")
                continue
        
        if valid_samples < samples * 0.8:
            logger.error(f"⚠️ Low valid sample rate: {valid_samples}/{samples}")
            logger.error("Calibration may be inaccurate. Check connections.")
        
        # Calculate offsets
        if valid_samples > 0:
            self.gyro_x_offset = sum_gx / valid_samples
            self.gyro_y_offset = sum_gy / valid_samples
            self.gyro_z_offset = sum_gz / valid_samples
            
            self.accel_x_offset = sum_ax / valid_samples
            self.accel_y_offset = sum_ay / valid_samples
            self.accel_z_offset = (sum_az / valid_samples) - 16384
            
            logger.info(f"✅ Calibration Complete! ({valid_samples} valid samples)")
            logger.info(f"   Gyro Offsets: X={self.gyro_x_offset:.1f}, "
                       f"Y={self.gyro_y_offset:.1f}, Z={self.gyro_z_offset:.1f}")
            logger.info(f"   Accel Offsets: X={self.accel_x_offset:.1f}, "
                       f"Y={self.accel_y_offset:.1f}, Z={self.accel_z_offset:.1f}")
        else:
            logger.error("❌ Calibration failed - no valid samples!")
    
    def _dynamic_recalibrate(self):
        """Auto recalibrate when robot is stationary"""
        if len(self.motion_history) < 50:
            return
        
        avg_motion = sum(self.motion_history) / len(self.motion_history)
        
        if avg_motion < self.STATIONARY_THRESHOLD:
            current_time = time.time()
            
            if current_time - self.last_recalib_time > self.RECALIB_INTERVAL:
                logger.info("🔄 Auto-recalibrating (robot stationary)...")
                
                sum_gz = 0
                valid = 0
                
                for _ in range(50):
                    try:
                        gz = self._read_word_safe(self.GYRO_ZOUT_H)
                        if gz != 0:
                            sum_gz += gz
                            valid += 1
                        time.sleep(0.01)
                    except:
                        continue
                
                if valid > 25:  # At least 50% success rate
                    new_offset = sum_gz / valid
                    self.gyro_z_offset = 0.9 * self.gyro_z_offset + 0.1 * new_offset
                    logger.info(f"✅ Recalibration done. New Z offset: {self.gyro_z_offset:.1f}")
                
                self.last_recalib_time = current_time
    
    def start(self):
        """Start sensor update thread"""
        if not self.connected:
            logger.error("❌ Cannot start: IMU not connected")
            return False
        
        if self.running:
            logger.warning("⚠️ IMU already running")
            return True
        
        self.running = True
        self.prev_time = time.time()
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        
        logger.info("✅ IMU sensor started")
        return True
    
    def stop(self):
        """Stop sensor update thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        logger.info("🛑 IMU sensor stopped")
    
    def _update_loop(self):
        """Main sensor fusion loop (100Hz)"""
        logger.info("🔄 IMU update loop started")
        
        consecutive_errors = 0
        MAX_CONSECUTIVE_ERRORS = 10
        
        while self.running and self.connected:
            try:
                current_time = time.time()
                dt = current_time - self.prev_time
                self.prev_time = current_time
                
                if dt <= 0 or dt > 0.1:
                    dt = 0.01
                
                # ===== READ GYRO =====
                gyro_x_raw = self._read_word_safe(self.GYRO_XOUT_H)
                gyro_y_raw = self._read_word_safe(self.GYRO_YOUT_H)
                gyro_z_raw = self._read_word_safe(self.GYRO_ZOUT_H)
                
                gyro_x = (gyro_x_raw - self.gyro_x_offset) / 131.0
                gyro_y = (gyro_y_raw - self.gyro_y_offset) / 131.0
                gyro_z = (gyro_z_raw - self.gyro_z_offset) / 131.0
                
                # Temperature compensation
                current_temp = self._read_temperature_safe()
                temp_drift = (current_temp - self.temp_baseline) * self.temp_drift_coeff
                gyro_z -= temp_drift
                
                # Track motion
                motion_magnitude = math.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
                self.motion_history.append(motion_magnitude)
                
                # ===== READ ACCELEROMETER =====
                accel_x_raw = self._read_word_safe(self.ACCEL_XOUT_H)
                accel_y_raw = self._read_word_safe(self.ACCEL_YOUT_H)
                accel_z_raw = self._read_word_safe(self.ACCEL_ZOUT_H)
                
                accel_x = (accel_x_raw - self.accel_x_offset) / 16384.0
                accel_y = (accel_y_raw - self.accel_y_offset) / 16384.0
                accel_z = (accel_z_raw - self.accel_z_offset) / 16384.0
                
                # ===== CALCULATE ANGLES =====
                accel_roll = math.atan2(accel_y, accel_z) * 180 / math.pi
                accel_pitch = math.atan2(-accel_x, math.sqrt(accel_y**2 + accel_z**2)) * 180 / math.pi
                
                # ===== COMPLEMENTARY FILTER =====
                gyro_roll = self.roll + gyro_x * dt
                self.roll = self.ALPHA * gyro_roll + (1 - self.ALPHA) * accel_roll
                
                gyro_pitch = self.pitch + gyro_y * dt
                self.pitch = self.ALPHA * gyro_pitch + (1 - self.ALPHA) * accel_pitch
                
                # Yaw (Gyro only)
                if abs(gyro_z) > 0.5:
                    self.yaw += gyro_z * dt
                
                # Keep Yaw in [-180, 180]
                if self.yaw > 180:
                    self.yaw -= 360
                elif self.yaw < -180:
                    self.yaw += 360
                
                # Dynamic recalibration
                self._dynamic_recalibrate()
                
                # Reset error counter on success
                consecutive_errors = 0
                
                time.sleep(0.01)  # 100Hz
                
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"❌ Error in update loop: {e}")
                
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    logger.error("⚠️ Too many consecutive errors! Stopping IMU.")
                    self.connected = False
                    break
                
                time.sleep(0.1)
        
        logger.info("🛑 IMU update loop ended")
    
    # ===== PUBLIC INTERFACE (Giữ nguyên) =====
    
    def get_yaw(self):
        return self.yaw
    
    def get_roll(self):
        return self.roll
    
    def get_pitch(self):
        return self.pitch
    
    def get_orientation(self):
        return {
            'roll': self.roll,
            'pitch': self.pitch,
            'yaw': self.yaw,
            'connected': self.connected
        }
    
    def reset_yaw(self):
        self.yaw = 0.0
        logger.info("🔄 Yaw reset to 0°")
    
    def reset_all(self):
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        logger.info("🔄 All angles reset")
    
    def is_level(self, tolerance=5.0):
        return abs(self.roll) < tolerance and abs(self.pitch) < tolerance
    
    def get_status(self):
        return {
            'connected': self.connected,
            'running': self.running,
            'roll': self.roll,
            'pitch': self.pitch,
            'yaw': self.yaw,
            'temperature': self._read_temperature_safe(),
            'is_level': self.is_level(),
            'last_read_age': time.time() - self.last_successful_read,
            'read_error_count': self.read_error_count
        }


# ===== TESTING =====
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*60)
    print("MPU-6050 Sensor Fusion Test (FIXED VERSION)")
    print("="*60 + "\n")
    
    imu = IMUSensorFusion()
    
    if not imu.connected:
        print("❌ Failed to connect to IMU!")
        print("\n🔧 Troubleshooting:")
        print("  1. Run: sudo i2cdetect -y 1 (must see 68)")
        print("  2. Check wiring")
        print("  3. Verify 3.3V power")
        sys.exit(1)
    
    if not imu.start():
        print("❌ Failed to start IMU!")
        sys.exit(1)
    
    print("\n📊 Monitoring IMU (Press Ctrl+C to stop)...")
    print("Tip: Tilt robot to see Roll/Pitch, rotate to see Yaw\n")
    
    try:
        while True:
            status = imu.get_status()
            
            if not status['connected']:
                print("\n❌ IMU disconnected!")
                break
            
            print(f"\r"
                  f"Roll: {status['roll']:7.2f}°  |  "
                  f"Pitch: {status['pitch']:7.2f}°  |  "
                  f"Yaw: {status['yaw']:7.2f}°  |  "
                  f"Temp: {status['temperature']:.1f}°C  |  "
                  f"Errors: {status['read_error_count']}  |  "
                  f"Level: {'✅' if status['is_level'] else '❌'}",
                  end="", flush=True)
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping IMU...")
        imu.stop()
        print("✅ Test completed\n")