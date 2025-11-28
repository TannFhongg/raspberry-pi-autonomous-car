"""
Robot Controller - Interface between Web Dashboard and Motor Driver
Handles commands from Flask app and controls motors
Updated with PID-based Auto Mode, YOLOv11 NCNN and Picamera2 support
"""

import threading
import time
import logging
import numpy as np
from typing import Optional
from datetime import datetime


# Import PID, lane detection and Object Detector
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from control.pid_controller import PIDController
from perception.lane_detector import detect_line
# Import get_web_camera để dùng chung camera với Web
from perception.camera_manager import CameraManager, get_web_camera 
from perception.object_detector import ObjectDetector

from perception.imu_sensor import IMUSensor
from perception.visual_odometry import VisualOdometry

logger = logging.getLogger(__name__)


class RobotController:
    """
    Main robot controller
    Manages motor control, safety, and state
    """
    
    def __init__(self, motor_driver, config: dict):
        self.driver = motor_driver
        self.config = config
        
        # Current state
        self.current_mode = 'manual'
        self.current_state = 'IDLE'
        self.current_speed = 180
        
        # Safety
        self.emergency_stopped = False
        self.last_command_time = time.time()
        self.timeout = config.get('safety', {}).get('timeout', 5.0)
        
        # Watchdog thread
        self.running = True
        self.watchdog_thread = threading.Thread(target=self._watchdog, daemon=True)
        self.watchdog_thread.start()

        # Khởi tạo IMU
        self.imu = IMUSensor()
        self.imu.start()

        # Khởi tạo Visual Odometry
        self.vo = VisualOdometry()
        
        logger.info("Robot Controller initialized")
    
    def smart_turn(self, target_angle: float, speed: int = 220, timeout: float = 5.0):
        """
        Rẽ chính xác sử dụng IMU (Closed-loop control)
        
        Args:
            target_angle: Góc cần rẽ (Độ). 
                          +90 = Rẽ Trái 90 độ
                          -90 = Rẽ Phải 90 độ
            speed: Tốc độ động cơ khi rẽ (mặc định 220)
            timeout: Thời gian tối đa cho phép (để tránh xe quay mãi nếu kẹt)
        """
        # Kiểm tra xem đã có IMU chưa
        if not hasattr(self, 'imu') or self.imu is None:
            logger.warning("⚠️ IMU chưa được khởi tạo! Chuyển sang rẽ theo thời gian (Fallback).")
            # Fallback: Rẽ mù theo thời gian (ước lượng: 220 ~ 0.6s cho 90 độ)
            duration = 0.6 * (abs(target_angle) / 90.0)
            if target_angle > 0: self.driver.turn_left(speed)
            else: self.driver.turn_right(speed)
            time.sleep(duration)
            self.driver.stop()
            return

        logger.info(f"🔄 Smart Turn START: Target {target_angle}°")
        
        # 1. Reset góc hiện tại về 0 để bắt đầu tính
        self.imu.reset_yaw()
        
        start_time = time.time()
        
        while True:
            # Lấy góc hiện tại từ IMU
            current_yaw = self.imu.get_yaw()
            
            # Tính sai số (còn phải quay bao nhiêu độ nữa?)
            # abs() để tính độ lớn, không quan tâm dấu
            error = abs(target_angle) - abs(current_yaw)
            
            # --- ĐIỀU KIỆN DỪNG ---
            
            # A. Đã đạt mục tiêu (Sai số < 2 độ)
            if error <= 2.0:
                logger.info(f"✅ Target Reached! Final Yaw: {current_yaw:.1f}°")
                break
            
            # B. Hết thời gian (Timeout) - Tránh treo vòng lặp
            if time.time() - start_time > timeout:
                logger.warning(f"⚠️ Turn Timeout! Stopped at {current_yaw:.1f}°")
                break

            # --- ĐIỀU KHIỂN TỐC ĐỘ (PID Đơn giản) ---
            # Giảm tốc độ khi gần đến đích để dừng chính xác hơn
            
            if error > 30: 
                # Còn xa > 30 độ: Chạy tốc độ cao
                current_speed = speed
            elif error > 10:
                # Gần đến nơi (10-30 độ): Giảm còn 70%
                current_speed = int(speed * 0.7)
            else:
                # Rất gần (< 10 độ): Giảm còn 50% (nhưng không dưới 130 để đủ lực thắng ma sát)
                current_speed = max(130, int(speed * 0.5))

            # --- GỬI LỆNH ĐỘNG CƠ ---
            if target_angle > 0:
                # Target dương -> Rẽ Trái (Yaw tăng)
                # Nếu lỡ quay lố (Overshoot) -> Rẽ phải nhẹ lại (tùy chọn, ở đây ta chỉ dừng)
                if current_yaw > target_angle: 
                    break 
                self.driver.turn_left(current_speed)
            else:
                # Target âm -> Rẽ Phải (Yaw giảm)
                if current_yaw < target_angle: 
                    break
                self.driver.turn_right(current_speed)
            
            # Ngủ cực ngắn để không chiếm hết CPU
            time.sleep(0.01)

        # Dừng động cơ ngay lập tức
        self.driver.stop()
        
        # Dừng thêm 0.2s để xe ổn định hẳn trước khi làm việc tiếp
        time.sleep(0.2)
    
    def set_mode(self, mode: str):
        if mode in ['manual', 'auto', 'follow']:
            self.current_mode = mode
            if mode == 'auto': self.current_state = 'AUTO MODE'
            elif mode == 'follow': self.current_state = 'FOLLOW MODE'
            else: self.current_state = 'IDLE'
            logger.info(f"Mode changed to: {mode}")
            return True
        return False
    
    def set_speed(self, speed: int):
        self.current_speed = max(0, min(255, speed))
        logger.info(f"Speed set to: {self.current_speed}")
    
    def forward(self):
        if not self._check_manual_mode(): return False
        self.driver.forward(self.current_speed)
        self.current_state = 'MOVING FORWARD'
        self._update_command_time()
        return True
    
    def backward(self):
        if not self._check_manual_mode(): return False
        self.driver.backward(self.current_speed)
        self.current_state = 'MOVING BACKWARD'
        self._update_command_time()
        return True
    
    def left(self):
        if not self._check_manual_mode(): return False
        turn_speed = int(self.current_speed * 0.8)
        self.driver.turn_left(turn_speed)
        self.current_state = 'TURNING LEFT'
        self._update_command_time()
        return True
    
    def right(self):
        if not self._check_manual_mode(): return False
        turn_speed = int(self.current_speed * 0.8)
        self.driver.turn_right(turn_speed)
        self.current_state = 'TURNING RIGHT'
        self._update_command_time()
        return True
    
    def stop(self):
        self.driver.stop()
        if self.current_mode == 'manual': self.current_state = 'STOPPED'
        elif self.current_mode == 'auto': self.current_state = 'AUTO MODE'
        elif self.current_mode == 'follow': self.current_state = 'FOLLOW MODE'
        self._update_command_time()
        return True
    
    def emergency_stop(self):
        self.driver.stop()
        self.emergency_stopped = True
        self.current_state = 'EMERGENCY STOP'
        logger.warning("EMERGENCY STOP ACTIVATED")
        return True
    
    def reset_emergency(self):
        self.emergency_stopped = False
        self.current_state = 'IDLE'
        logger.info("Emergency stop reset")
    
    def get_state(self) -> dict:
        left_speed, right_speed = self.driver.get_speeds()
        return {
            'mode': self.current_mode,
            'state': self.current_state,
            'speed': self.current_speed,
            'emergency_stopped': self.emergency_stopped,
            'left_motor_speed': left_speed,
            'right_motor_speed': right_speed,
            'last_command_age': time.time() - self.last_command_time
        }
    
    def _check_manual_mode(self) -> bool:
        if self.emergency_stopped:
            logger.warning("Cannot execute command: Emergency stop active")
            return False
        if self.current_mode != 'manual':
            logger.warning(f"Cannot execute command: Not in manual mode (current: {self.current_mode})")
            return False
        return True
    
    def _update_command_time(self):
        self.last_command_time = time.time()
    
    def _watchdog(self):
        while self.running:
            time.sleep(0.5)
            age = time.time() - self.last_command_time
            if age > self.timeout and self.current_mode == 'manual':
                left, right = self.driver.get_speeds()
                if left != 0 or right != 0:
                    logger.warning(f"Command timeout ({age:.1f}s) - Auto stopping motors")
                    self.stop()
            if self.current_state in ['MOVING FORWARD', 'MOVING BACKWARD', 'TURNING LEFT', 'TURNING RIGHT']:
                left, right = self.driver.get_speeds()
                if left == 0 and right == 0:
                    self.current_state = 'IDLE'
    
    def cleanup(self):
        self.running = False
        self.driver.cleanup()
        logger.info("Robot Controller cleaned up")


class AutoModeController:
    """
    Autonomous mode controller
    Combines Lane Following (PID) and Traffic Sign Recognition (YOLOv11)
    """
    
    def __init__(self, robot_controller: RobotController):
        self.robot = robot_controller
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.camera: Optional[CameraManager] = None
        
        # AI Detector
        self.detector = ObjectDetector(model_path='data/models/best_ncnn_model', conf_threshold=0.5)
        
        # PID
        pid_config = robot_controller.config.get('lane_following', {}).get('pid', {})
        self.pid = PIDController(
            kp=pid_config.get('kp', 0.8),
            ki=pid_config.get('ki', 0.0),
            kd=pid_config.get('kd', 0.3),
            output_min=pid_config.get('min_output', -255),
            output_max=pid_config.get('max_output', 255),
            derivative_smoothing=pid_config.get('derivative_smoothing', 0.7)
        )
        
        # Lane settings
        lane_config = robot_controller.config.get('lane_following', {})
        self.base_speed = lane_config.get('base_speed', 150)
        self.default_speed = self.base_speed
        self.max_speed = lane_config.get('max_speed', 255)
        self.min_speed = lane_config.get('min_speed', 60)
        self.detection_config = robot_controller.config.get('ai', {}).get('lane_detection', {})
        
        # --- CẤU HÌNH KHOẢNG CÁCH BIỂN BÁO (140px - 170px) ---
        self.DIST_PREPARE = 140  # < 140px: Chưa làm gì
        self.DIST_EXECUTE = 170  # 140px - 170px: Vùng hành động
        
        # State
        self.lane_lost_count = 0
        self.lane_lost_threshold = 10
        self.latest_debug_frame = None
        self.latest_error = 0
        self.latest_correction = 0
        
        logger.info("Auto Mode Controller initialized with YOLOv11 & Distance Logic")
    
    def start(self):
        if not self.running:
            if not self._init_shared_camera(): return False
            
            self.pid.reset()
            self.lane_lost_count = 0
            self.base_speed = self.default_speed
            
            self.running = True
            self.thread = threading.Thread(target=self._auto_loop, daemon=True)
            self.thread.start()
            logger.info("Auto mode started")
            return True
        return False
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        self.robot.driver.stop()
        logger.info("Auto mode stopped")
    
    def _init_shared_camera(self) -> bool:
        try:
            self.camera = get_web_camera(self.robot.config)
            if not self.camera.is_running():
                if not self.camera.start(): return False
            return True
        except Exception as e:
            logger.error(f"Camera init error: {e}")
            return False
    
    def _auto_loop(self):
        logger.info("Auto loop started")
        
        while self.running:
            try:
                if self.robot.current_mode != 'auto': break
                
                frame = self.camera.capture_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                detections, debug_frame = self.detector.detect(frame)
                sign_action = None
                
                if detections:
                    sign = max(detections, key=lambda x: x['w'] * x['h'])
                    sign_name = sign['class_name']
                    sign_size = max(sign['w'], sign['h'])
                    
                    self.robot.current_state = f"SIGN: {sign_name} ({sign_size:.0f}px)"
                    
                    # --- LOGIC KHOẢNG CÁCH ---
                    
                    if sign_size < self.DIST_PREPARE:
                        pass # Chưa làm gì
                    
                    elif sign_size > self.DIST_EXECUTE + 20:
                        pass # Đã đi qua
                    
                    else:
                        # VÙNG HÀNH ĐỘNG (140px - 170px)
                        logger.info(f"EXECUTING ACTION FOR: {sign_name} (Size: {sign_size:.0f})")
                        
                        if sign_name in ['stop_sign', 'red_light']:
                            self.robot.driver.stop()
                            sign_action = "STOP"
                            time.sleep(0.1)
                            
                        elif sign_name == 'green_light':
                            self.base_speed = self.default_speed
                            pass
                            
                        elif sign_name == 'left_turn_sign':
                            # RẼ TRÁI: Tốc độ 220 (Mạnh)
                            loger.info("Detected Left Turn Sign -> Smart Turn 90")
                            self.robot.smart_turn(90, speed=220) 
                            continue
                            
                            
                        elif sign_name == 'right_turn_sign':
                            logger.info("Detected Left Turn Sign -> Smart Turn 90")
                        # Gọi hàm rẽ thông minh: 90 độ, tốc độ 220
                            self.robot.smart_turn(-90, speed=220) 
                            continue
                            
                        elif sign_name == 'speed_limit_signs':
                            self.base_speed = 100
                        
                        elif sign_name == 'parking_signs':
                            self.robot.driver.stop()
                            self.stop()
                            break

                if sign_action in ["STOP", "TURN"]:
                    continue

                # Lane Following
                error, x_line, center_x, lane_debug_frame = detect_line(frame, self.detection_config)
                self.latest_debug_frame = lane_debug_frame 
                self.latest_error = error
                
                # Tính PID & Điều khiển
                # ... (Đoạn này giữ nguyên như logic chuẩn)
                current_time = time.time()
                dt = 0.05 # Ước lượng dt
                
                if abs(error) > frame.shape[1] * 0.4:
                    self.lane_lost_count += 1
                    if self.lane_lost_count >= self.lane_lost_threshold:
                        self.robot.driver.stop()
                        self.robot.current_state = 'LANE LOST'
                        continue
                else:
                    self.lane_lost_count = 0
                    if not detections: self.robot.current_state = 'FOLLOWING LANE'
                
                correction = self.pid.compute(error, dt)
                self.latest_correction = correction
                
                left_speed = max(-255, min(255, int(self.base_speed - correction)))
                right_speed = max(-255, min(255, int(self.base_speed + correction)))
                
                self.robot.driver.set_motors(left_speed, right_speed)
                time.sleep(0.03)
                
            except Exception as e:
                logger.error(f"Error in auto loop: {e}")
                self.robot.driver.stop()
                break
        
        self.robot.driver.stop()
        logger.info("Auto loop ended")

    def get_debug_frame(self):
        return self.latest_debug_frame
    
    def get_pid_status(self):
        return {'error': self.latest_error, 'correction': self.latest_correction, **self.pid.get_components()}


class FollowModeController:
    """
    Follow mode controller
    Uses YOLOv11 to track specific colored objects (6cm x 6cm targets)
    """
    
    def __init__(self, robot_controller: RobotController):
        self.robot = robot_controller
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.camera: Optional[CameraManager] = None
        
        # AI Detector
        self.detector = ObjectDetector(model_path='data/models/best_ncnn_model', conf_threshold=0.5)
        
        self.color_map = {
            'red': 'red_color',
            'green': 'green_color',
            'blue': 'blue_color',
            'yellow': 'yellow_color'
        }
        self.target_color_name = 'red'
        self.pid_turn = PIDController(kp=0.6, ki=0.0, kd=0.2, output_max=255)
        
        # --- CẤU HÌNH CAMERA (Calibration) ---
        self.FOCAL_LENGTH = 542  # Tiêu cự (đã tính ở bài trước)
        self.OBJECT_WIDTH = 6    # Kích thước vật thể (cm)
        
        # Khởi tạo các ngưỡng khoảng cách mặc định (50cm)
        self.SIZE_FORWARD = 0
        self.SIZE_STOP = 0
        self.SIZE_BACK = 0
        self.set_follow_distance(50) # Cài đặt mặc định ban đầu
        
        # Web Info
        self.target_x = 0
        self.target_y = 0
        self.target_w = 0
        self.target_h = 0
        self.confidence = 0
        self.target_distance = 0
        
        logger.info("Follow Mode Controller initialized with YOLO")
    
    def start(self):
        if not self.running:
            if not self._init_shared_camera(): return False
            self.running = True
            self.thread = threading.Thread(target=self._follow_loop, daemon=True)
            self.thread.start()
            logger.info(f"Follow mode started: {self.target_color_name}")
            return True
        return False
    
    def stop(self):
        self.running = False
        if self.thread: self.thread.join(timeout=2.0)
        self.robot.driver.stop()
        logger.info("Follow mode stopped")
    
    def set_target_color(self, color: str):
        self.target_color_name = color
        logger.info(f"Target color changed to: {color}")
    
    def set_follow_distance(self, distance: int):
        """
        Cập nhật khoảng cách bám theo từ Web Dashboard
        Input: distance (cm)
        Output: Cập nhật các ngưỡng Pixel (SIZE_STOP...)
        """
        if distance < 10: distance = 10 # Giới hạn tối thiểu 10cm
        
        # Tính kích thước Pixel mục tiêu tại khoảng cách đó
        # Công thức: Pixel = (Focal * Real_Size) / Distance
        target_pixel_size = (self.FOCAL_LENGTH * self.OBJECT_WIDTH) / distance
        
        # Thiết lập các vùng hành động xung quanh kích thước mục tiêu
        # Ví dụ: Nếu muốn dừng ở 100px -> Tiến khi < 90, Lùi khi > 110
        margin = 10 # Khoảng đệm
        
        self.SIZE_STOP = int(target_pixel_size)      # Điểm dừng chuẩn
        self.SIZE_FORWARD = int(target_pixel_size - margin) # Xa hơn -> Tiến
        self.SIZE_BACK = int(target_pixel_size + margin)    # Gần hơn -> Lùi
        
        logger.info(f"Set Follow Distance: {distance}cm -> Stop Size: {self.SIZE_STOP}px")
    
    def get_target_data(self) -> dict:
        # Tính ngược lại khoảng cách hiện tại để hiển thị lên Web
        current_dist = 0
        if self.target_w > 0:
            current_dist = (self.FOCAL_LENGTH * self.OBJECT_WIDTH) / self.target_w
            
        return {
            'tracking': self.confidence > 0,
            'target_color': self.target_color_name,
            'target_x': self.target_x,
            'target_y': self.target_y,
            'target_w': self.target_w,
            'target_h': self.target_h,
            'confidence': self.confidence,
            'target_distance': current_dist # Gửi khoảng cách thật về Web
        }
    
    def _init_shared_camera(self) -> bool:
        try:
            self.camera = get_web_camera(self.robot.config)
            if not self.camera.is_running():
                if not self.camera.start(): return False
            return True
        except Exception as e:
            logger.error(f"Camera init error: {e}")
            return False

    def _follow_loop(self):
        logger.info(f"Follow loop started. Tracking: {self.color_map.get(self.target_color_name)}")
        
        while self.running:
            try:
                if self.robot.current_mode != 'follow': break
                
                frame = self.camera.capture_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                detections, _ = self.detector.detect(frame)
                target_class = self.color_map.get(self.target_color_name)
                valid_objs = [d for d in detections if d['class_name'] == target_class]
                
                if valid_objs:
                    target = max(valid_objs, key=lambda x: x['w'] * x['h'])
                    
                    # PID Rẽ
                    center_x = frame.shape[1] / 2
                    error_x = center_x - target['x']
                    turn_output = self.pid_turn.compute(error_x)
                    
                    # Điều khiển Tốc độ (Dựa trên các ngưỡng đã tính động)
                    # Lấy cạnh lớn nhất để ổn định (vì hình vuông 6x6)
                    obj_size = max(target['w'], target['h'])
                    
                    if obj_size < self.SIZE_FORWARD:
                        forward_speed = 220 # Xa -> Tiến nhanh
                    elif obj_size > self.SIZE_BACK:
                        forward_speed = -150 # Quá gần -> Lùi
                    elif obj_size > self.SIZE_STOP:
                         forward_speed = 0 # Hơi gần -> Dừng
                    else:
                        forward_speed = 0 # Đúng tầm -> Dừng
                    
                    left_speed = max(-255, min(255, int(forward_speed + turn_output)))
                    right_speed = max(-255, min(255, int(forward_speed - turn_output)))
                    
                    self.robot.driver.set_motors(left_speed, right_speed)
                    self.robot.current_state = f"TRACKING {target['class_name']} ({obj_size:.0f}px)"
                    
                    self.target_x = int(target['x'])
                    self.target_y = int(target['y'])
                    self.target_w = int(target['w'])
                    self.target_h = int(target['h'])
                    self.confidence = int(target['conf'] * 100)
                    
                else:
                    self.robot.driver.stop()
                    self.robot.current_state = "SEARCHING..."
                    self.confidence = 0
                    self.target_w = 0 # Reset width để tính distance = 0
                
                time.sleep(0.05)
                
            except Exception as e:
                logger.error(f"Follow loop error: {e}")
                self.robot.driver.stop()
                break
        
        self.robot.driver.stop()
        logger.info("Follow loop ended")