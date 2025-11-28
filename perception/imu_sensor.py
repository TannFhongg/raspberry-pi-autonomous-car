import smbus2
import math
import time
import threading

class IMUSensor:
    def __init__(self, address=0x68, bus_num=1):
        self.bus = smbus2.SMBus(bus_num)
        self.address = address
        self.yaw = 0.0
        self.prev_time = time.time()
        self.running = False
        
        # Đánh thức MPU-6050 (Ghi 0 vào thanh ghi Power Management 1)
        try:
            self.bus.write_byte_data(self.address, 0x6B, 0)
            # Cấu hình Gyro (ví dụ: +/- 250 deg/s)
            self.bus.write_byte_data(self.address, 0x1B, 0)
            self.connected = True
        except Exception as e:
            print(f"IMU Error: {e}")
            self.connected = False

        # Calibration (Tính sai số tĩnh lúc đứng yên)
        self.gyro_z_offset = 0.0
        if self.connected:
            self._calibrate()

    def _read_word(self, reg):
        h = self.bus.read_byte_data(self.address, reg)
        l = self.bus.read_byte_data(self.address, reg + 1)
        val = (h << 8) + l
        if val >= 0x8000: return -((65535 - val) + 1)
        return val

    def _calibrate(self, samples=100):
        print("Calibrating IMU... DO NOT MOVE!")
        sum_z = 0
        for _ in range(samples):
            sum_z += self._read_word(0x47) # 0x47 là thanh ghi Gyro Z
            time.sleep(0.01)
        self.gyro_z_offset = sum_z / samples
        print(f"IMU Calibrated. Offset Z: {self.gyro_z_offset}")

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def _update_loop(self):
        while self.running and self.connected:
            current_time = time.time()
            dt = current_time - self.prev_time
            self.prev_time = current_time
            
            # Đọc Gyro Z
            gyro_z_raw = self._read_word(0x47)
            gyro_z = (gyro_z_raw - self.gyro_z_offset) / 131.0 # 131 LSB/(deg/s)
            
            # Tích phân để tính góc (Yaw)
            # Nếu giá trị quá nhỏ (nhiễu), bỏ qua
            if abs(gyro_z) > 0.5:
                self.yaw += gyro_z * dt
            
            time.sleep(0.01) # 100Hz

    def get_yaw(self):
        return self.yaw
    
    def reset_yaw(self):
        self.yaw = 0.0