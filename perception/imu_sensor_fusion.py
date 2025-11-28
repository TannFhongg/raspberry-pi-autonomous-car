"""
IMU Sensor with Complementary Filter (Production Ready)
✅ Gyro + Accelerometer fusion
✅ Drift reduction for Roll/Pitch
✅ Temperature compensation
✅ Dynamic recalibration
✅ Connection monitoring
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
    MPU-6050 với Complementary Filter
    Kết hợp Gyro (smooth, có drift) + Accel (không drift, nhiễu)
    """
    
    # MPU-6050 Register Map
    PWR_MGMT_1 = 0x6B
    GYRO_CONFIG = 0x1B
    ACCEL_CONFIG = 0x1C
    TEMP_OUT_H = 0x41
    
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
        self.roll = 0.0   # X-axis rotation (Fused)
        self.pitch = 0.0  # Y-axis rotation (Fused)
        self.yaw = 0.0    # Z-axis rotation (Gyro only - still drifts!)
        
        # Complementary Filter coefficient
        # 0.98 = 98% trust Gyro (short-term), 2% trust Accel (long-term)
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
        self.accel_z_offset = 1.0  # Gravity baseline
        
        # Temperature compensation
        self.temp_baseline = 0.0
        self.temp_drift_coeff = 0.01  # deg/°C (empirical)
        
        # Dynamic recalibration (Detect when robot is stationary)
        self.motion_history = deque(maxlen=50)  # Last 0.5s of motion
        self.STATIONARY_THRESHOLD = 0.5  # deg/s
        self.last_recalib_time = time.time()
        self.RECALIB_INTERVAL = 30.0  # Auto recalibrate every 30s when stationary
        
        # Connection monitoring
        self.last_successful_read = time.time()
        self.CONNECTION_TIMEOUT = 5.0
        
        # Initialize hardware
        self._init_hardware(bus_num)
    
    def _init_hardware(self, bus_num):
        """Initialize MPU-6050"""
        try:
            self.bus = smbus2.SMBus(bus_num)
            
            # Wake up MPU-6050 (Write 0 to PWR_MGMT_1)
            self.bus.write_byte_data(self.address, self.PWR_MGMT_1, 0)
            time.sleep(0.1)
            
            # Configure Gyro: ±250 deg/s (Most sensitive)
            self.bus.write_byte_data(self.address, self.GYRO_CONFIG, 0)
            
            # Configure Accel: ±2g (Most sensitive)
            self.bus.write_byte_data(self.address, self.ACCEL_CONFIG, 0)
            
            # Read baseline temperature
            self.temp_baseline = self._read_temperature()
            
            self.connected = True
            logger.info("✅ MPU-6050 initialized successfully")
            
            # Calibrate
            self._calibrate()
            
        except Exception as e:
            logger.error(f"❌ MPU-6050 initialization failed: {e}")
            self.connected = False
    
    def _read_word(self, reg):
        """Read signed 16-bit word from register"""
        try:
            h = self.bus.read_byte_data(self.address, reg)
            l = self.bus.read_byte_data(self.address, reg + 1)
            val = (h << 8) + l
            
            # Convert to signed
            if val >= 0x8000:
                return -((65535 - val) + 1)
            
            self.last_successful_read = time.time()
            return val
            
        except Exception as e:
            logger.error(f"❌ Read error at register {hex(reg)}: {e}")
            self._check_connection()
            return 0
    
    def _read_temperature(self):
        """Read internal temperature (°C)"""
        try:
            temp_raw = self._read_word(self.TEMP_OUT_H)
            # Formula from datasheet: Temp = (TEMP_OUT / 340) + 36.53
            temp_c = (temp_raw / 340.0) + 36.53
            return temp_c
        except:
            return self.temp_baseline
    
    def _check_connection(self):
        """Monitor connection health"""
        if time.time() - self.last_successful_read > self.CONNECTION_TIMEOUT:
            logger.error("❌ IMU connection lost!")
            self.connected = False
    
    def _calibrate(self, samples=500):
        """
        Calibrate gyro and accelerometer offsets
        Robot MUST be stationary on flat surface!
        """
        if not self.connected:
            return
        
        logger.info("🔧 Calibrating IMU... KEEP ROBOT STILL AND LEVEL!")
        
        sum_gx, sum_gy, sum_gz = 0, 0, 0
        sum_ax, sum_ay, sum_az = 0, 0, 0
        
        for i in range(samples):
            # Read raw values
            sum_gx += self._read_word(self.GYRO_XOUT_H)
            sum_gy += self._read_word(self.GYRO_YOUT_H)
            sum_gz += self._read_word(self.GYRO_ZOUT_H)
            
            sum_ax += self._read_word(self.ACCEL_XOUT_H)
            sum_ay += self._read_word(self.ACCEL_YOUT_H)
            sum_az += self._read_word(self.ACCEL_ZOUT_H)
            
            if i % 100 == 0:
                logger.info(f"Calibration progress: {i}/{samples}")
            
            time.sleep(0.01)
        
        # Calculate offsets
        self.gyro_x_offset = sum_gx / samples
        self.gyro_y_offset = sum_gy / samples
        self.gyro_z_offset = sum_gz / samples
        
        # Accel offsets (Z should be ~1g = 16384 for ±2g range)
        self.accel_x_offset = sum_ax / samples
        self.accel_y_offset = sum_ay / samples
        self.accel_z_offset = (sum_az / samples) - 16384  # Remove gravity
        
        logger.info(f"✅ Calibration Complete!")
        logger.info(f"   Gyro Offsets: X={self.gyro_x_offset:.1f}, "
                   f"Y={self.gyro_y_offset:.1f}, Z={self.gyro_z_offset:.1f}")
        logger.info(f"   Accel Offsets: X={self.accel_x_offset:.1f}, "
                   f"Y={self.accel_y_offset:.1f}, Z={self.accel_z_offset:.1f}")
    
    def _dynamic_recalibrate(self):
        """
        Auto recalibrate when robot is stationary
        Helps reduce long-term drift
        """
        if len(self.motion_history) < 50:
            return
        
        # Check if robot has been stationary
        avg_motion = sum(self.motion_history) / len(self.motion_history)
        
        if avg_motion < self.STATIONARY_THRESHOLD:
            current_time = time.time()
            
            if current_time - self.last_recalib_time > self.RECALIB_INTERVAL:
                logger.info("🔄 Auto-recalibrating (robot stationary)...")
                
                # Quick recalibration (50 samples)
                sum_gz = 0
                for _ in range(50):
                    sum_gz += self._read_word(self.GYRO_ZOUT_H)
                    time.sleep(0.01)
                
                new_offset = sum_gz / 50
                
                # Smooth transition (don't jump suddenly)
                self.gyro_z_offset = 0.9 * self.gyro_z_offset + 0.1 * new_offset
                
                self.last_recalib_time = current_time
                logger.info(f"✅ Recalibration done. New Z offset: {self.gyro_z_offset:.1f}")
    
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
        
        while self.running and self.connected:
            try:
                current_time = time.time()
                dt = current_time - self.prev_time
                self.prev_time = current_time
                
                # Prevent division by zero
                if dt <= 0 or dt > 0.1:
                    dt = 0.01
                
                # ===== READ GYRO =====
                gyro_x_raw = self._read_word(self.GYRO_XOUT_H)
                gyro_y_raw = self._read_word(self.GYRO_YOUT_H)
                gyro_z_raw = self._read_word(self.GYRO_ZOUT_H)
                
                # Convert to deg/s (131 LSB/(deg/s) for ±250 deg/s range)
                gyro_x = (gyro_x_raw - self.gyro_x_offset) / 131.0
                gyro_y = (gyro_y_raw - self.gyro_y_offset) / 131.0
                gyro_z = (gyro_z_raw - self.gyro_z_offset) / 131.0
                
                # Temperature compensation for Yaw
                current_temp = self._read_temperature()
                temp_drift = (current_temp - self.temp_baseline) * self.temp_drift_coeff
                gyro_z -= temp_drift
                
                # Track motion for dynamic recalibration
                motion_magnitude = math.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
                self.motion_history.append(motion_magnitude)
                
                # ===== READ ACCELEROMETER =====
                accel_x_raw = self._read_word(self.ACCEL_XOUT_H)
                accel_y_raw = self._read_word(self.ACCEL_YOUT_H)
                accel_z_raw = self._read_word(self.ACCEL_ZOUT_H)
                
                # Convert to g (16384 LSB/g for ±2g range)
                accel_x = (accel_x_raw - self.accel_x_offset) / 16384.0
                accel_y = (accel_y_raw - self.accel_y_offset) / 16384.0
                accel_z = (accel_z_raw - self.accel_z_offset) / 16384.0
                
                # ===== CALCULATE ANGLES FROM ACCELEROMETER =====
                # (Stable long-term, but noisy)
                accel_roll = math.atan2(accel_y, accel_z) * 180 / math.pi
                accel_pitch = math.atan2(-accel_x, 
                                        math.sqrt(accel_y**2 + accel_z**2)) * 180 / math.pi
                
                # ===== COMPLEMENTARY FILTER: Fuse Gyro + Accel =====
                # Roll (X-axis)
                gyro_roll = self.roll + gyro_x * dt
                self.roll = self.ALPHA * gyro_roll + (1 - self.ALPHA) * accel_roll
                
                # Pitch (Y-axis)
                gyro_pitch = self.pitch + gyro_y * dt
                self.pitch = self.ALPHA * gyro_pitch + (1 - self.ALPHA) * accel_pitch
                
                # Yaw (Z-axis) - GYRO ONLY (Still drifts, needs magnetometer to fix)
                if abs(gyro_z) > 0.5:  # Noise gate
                    self.yaw += gyro_z * dt
                
                # Keep Yaw in [-180, 180] range
                if self.yaw > 180:
                    self.yaw -= 360
                elif self.yaw < -180:
                    self.yaw += 360
                
                # ===== DYNAMIC RECALIBRATION =====
                self._dynamic_recalibrate()
                
                time.sleep(0.01)  # 100Hz
                
            except Exception as e:
                logger.error(f"❌ Error in update loop: {e}")
                time.sleep(0.1)
        
        logger.info("🛑 IMU update loop ended")
    
    # ===== PUBLIC INTERFACE =====
    
    def get_yaw(self):
        """Get Yaw angle (Z-axis rotation) - Note: Still has drift!"""
        return self.yaw
    
    def get_roll(self):
        """Get Roll angle (X-axis rotation) - Fused, minimal drift"""
        return self.roll
    
    def get_pitch(self):
        """Get Pitch angle (Y-axis rotation) - Fused, minimal drift"""
        return self.pitch
    
    def get_orientation(self):
        """Get all angles"""
        return {
            'roll': self.roll,
            'pitch': self.pitch,
            'yaw': self.yaw,
            'connected': self.connected
        }
    
    def reset_yaw(self):
        """Reset Yaw angle to 0"""
        self.yaw = 0.0
        logger.info("🔄 Yaw reset to 0°")
    
    def reset_all(self):
        """Reset all angles to 0"""
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        logger.info("🔄 All angles reset")
    
    def is_level(self, tolerance=5.0):
        """Check if robot is level (for safe operation)"""
        return abs(self.roll) < tolerance and abs(self.pitch) < tolerance
    
    def get_status(self):
        """Get comprehensive status"""
        return {
            'connected': self.connected,
            'running': self.running,
            'roll': self.roll,
            'pitch': self.pitch,
            'yaw': self.yaw,
            'temperature': self._read_temperature(),
            'is_level': self.is_level(),
            'last_read_age': time.time() - self.last_successful_read
        }


# ===== TESTING PROGRAM =====
if __name__ == "__main__":
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*60)
    print("MPU-6050 Sensor Fusion Test")
    print("="*60 + "\n")
    
    # Initialize
    imu = IMUSensorFusion()
    
    if not imu.connected:
        print("❌ Failed to connect to IMU!")
        sys.exit(1)
    
    # Start
    if not imu.start():
        print("❌ Failed to start IMU!")
        sys.exit(1)
    
    print("\n📊 Monitoring IMU (Press Ctrl+C to stop)...")
    print("Tip: Tilt robot to see Roll/Pitch, rotate to see Yaw\n")
    
    try:
        while True:
            status = imu.get_status()
            
            # Display
            print(f"\r"
                  f"Roll: {status['roll']:7.2f}°  |  "
                  f"Pitch: {status['pitch']:7.2f}°  |  "
                  f"Yaw: {status['yaw']:7.2f}°  |  "
                  f"Temp: {status['temperature']:.1f}°C  |  "
                  f"Level: {'✅' if status['is_level'] else '❌'}",
                  end="", flush=True)
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping IMU...")
        imu.stop()
        print("✅ Test completed\n")