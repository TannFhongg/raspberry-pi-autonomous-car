"""
Robot Controller - COMPLETE VERSION
✅ Manual Mode
✅ Auto Mode (Lane Following)
✅ IMPROVED Follow Mode (Size-based distance control with PID)
"""

import threading
import time
import logging
import numpy as np
from typing import Optional
from datetime import datetime

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from control.pid_controller import PIDController
from perception.lane_detector import detect_line
from perception.camera_manager import CameraManager, get_web_camera 
from perception.object_detector import ObjectDetector
from perception.imu_sensor_fusion import IMUSensorFusion
from perception.visual_odometry import VisualOdometry
import cv2

logger = logging.getLogger(__name__)


# ============================================================
# ROBOT CONTROLLER (Base Class)
# ============================================================

class RobotController:
    """Main robot controller"""
    
    def __init__(self, motor_driver, config: dict):
        self.driver = motor_driver
        self.config = config
        
        # Current state
        self.current_mode = 'manual'
        self.current_state = 'IDLE'
        self.current_speed = 90
        
        # Safety
        self.emergency_stopped = False
        self.last_command_time = time.time()
        self.timeout = config.get('safety', {}).get('timeout', 5.0)
        
        # Watchdog thread
        self.running = True
        self.watchdog_thread = threading.Thread(target=self._watchdog, daemon=True)
        self.watchdog_thread.start()

        # IMU initialization
        try:
            self.imu = IMUSensorFusion()
            if self.imu.connected:
                self.imu.start()
                logger.info("✅ IMU initialized successfully")
            else:
                logger.warning("⚠️ IMU not connected! Smart turn will use fallback mode.")
                self.imu = None
        except Exception as e:
            logger.error(f"❌ IMU initialization failed: {e}")
            self.imu = None

        # Visual Odometry (chạy ngầm, chỉ hoạt động khi manual mode)
        vo_config = config.get('visual_odometry', {})
        scale = vo_config.get('scale_factor', 0.05)
        self.vo = VisualOdometry(scale_factor=scale)
        self.vo_map = None  # Ảnh bản đồ trajectory
        self.vo_camera = None  # Camera riêng cho VO
        
        # Start VO background thread
        self.vo_thread = threading.Thread(target=self._vo_loop, daemon=True)
        self.vo_thread.start()
        logger.info("✅ Visual Odometry initialized (background thread)")

        logger.info("Robot Controller initialized")
    
    def smart_turn(self, target_angle: float, speed: int = 220, timeout: float = 5.0):
        """Smart turn using IMU"""
        if speed < 130:
            logger.warning(f"Speed {speed} too low, setting to 130")
            speed = 130
        elif speed > 255:
            logger.warning(f"Speed {speed} too high, setting to 255")
            speed = 255
        
        if abs(target_angle) > 180:
            logger.error(f"❌ Invalid angle: {target_angle}° (must be -180 to 180)")
            return
        
        if not hasattr(self, 'imu') or self.imu is None or not self.imu.connected:
            logger.warning("⚠️ IMU unavailable! Using time-based fallback.")
            self._fallback_turn(target_angle, speed)
            return
        
        logger.info(f"🔄 Smart Turn START: Target {target_angle}° at speed {speed}")
        
        try:
            self.imu.reset_yaw()
            start_time = time.time()
            last_yaw = 0.0
            stuck_counter = 0
            
            while True:
                current_yaw = self.imu.get_yaw()
                error = abs(target_angle) - abs(current_yaw)
                
                if error <= 2.0:
                    logger.info(f"✅ Target Reached! Final: {current_yaw:.1f}°")
                    break
                
                if time.time() - start_time > timeout:
                    logger.warning(f"⚠️ Turn Timeout! Stopped at {current_yaw:.1f}° (Target: {target_angle}°)")
                    break
                
                if abs(current_yaw - last_yaw) < 0.1:
                    stuck_counter += 1
                    if stuck_counter > 50:
                        logger.error(f"❌ Robot appears stuck! Emergency stop.")
                        break
                else:
                    stuck_counter = 0
                
                last_yaw = current_yaw
                
                if target_angle > 0 and current_yaw > target_angle + 5:
                    logger.warning(f"⚠️ Overshoot detected! {current_yaw:.1f}° > {target_angle}°")
                    break
                elif target_angle < 0 and current_yaw < target_angle - 5:
                    logger.warning(f"⚠️ Overshoot detected! {current_yaw:.1f}° < {target_angle}°")
                    break
                
                if error > 30:
                    current_speed = speed
                elif error > 10:
                    current_speed = int(speed * 0.7)
                else:
                    current_speed = max(130, int(speed * 0.5))
                
                if target_angle > 0:
                    self.driver.turn_left(current_speed)
                else:
                    self.driver.turn_right(current_speed)
                
                time.sleep(0.01)
            
        except Exception as e:
            logger.error(f"❌ Error during smart turn: {e}")
        
        finally:
            self.driver.stop()
            time.sleep(0.2)
    
    def _fallback_turn(self, target_angle: float, speed: int):
        """Fallback turn based on time"""
        duration = 0.6 * (abs(target_angle) / 90.0)
        logger.info(f"⏱️ Fallback Turn: {target_angle}° for {duration:.2f}s")
        
        if target_angle > 0:
            self.driver.turn_left(speed)
        else:
            self.driver.turn_right(speed)
        
        time.sleep(duration)
        self.driver.stop()
    
    def set_mode(self, mode: str):
        if mode in ['manual', 'auto', 'follow']:
            self.current_mode = mode
            if mode == 'auto':
                self.current_state = 'AUTO MODE'
            elif mode == 'follow':
                self.current_state = 'FOLLOW MODE'
            else:
                self.current_state = 'IDLE'
            logger.info(f"Mode changed to: {mode}")
            return True
        return False
    
    def set_speed(self, speed: int):
        self.current_speed = max(0, min(255, speed))
        logger.info(f"Speed set to: {self.current_speed}")
    
    def forward(self):
        if not self._check_manual_mode():
            return False
        self.driver.forward(self.current_speed)
        self.current_state = 'MOVING FORWARD'
        self._update_command_time()
        return True
    
    def backward(self):
        if not self._check_manual_mode():
            return False
        self.driver.backward(self.current_speed)
        self.current_state = 'MOVING BACKWARD'
        self._update_command_time()
        return True
    
    def left(self):
        if not self._check_manual_mode():
            return False
        turn_speed = int(self.current_speed * 0.8)
        self.driver.turn_left(turn_speed)
        self.current_state = 'TURNING LEFT'
        self._update_command_time()
        return True
    
    def right(self):
        if not self._check_manual_mode():
            return False
        turn_speed = int(self.current_speed * 0.8)
        self.driver.turn_right(turn_speed)
        self.current_state = 'TURNING RIGHT'
        self._update_command_time()
        return True
    
    def stop(self):
        self.driver.stop()
        if self.current_mode == 'manual':
            self.current_state = 'STOPPED'
        elif self.current_mode == 'auto':
            self.current_state = 'AUTO MODE'
        elif self.current_mode == 'follow':
            self.current_state = 'FOLLOW MODE'
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
        imu_status = "Connected" if (self.imu and self.imu.connected) else "Disconnected"
        
        return {
            'mode': self.current_mode,
            'state': self.current_state,
            'speed': self.current_speed,
            'emergency_stopped': self.emergency_stopped,
            'left_motor_speed': left_speed,
            'right_motor_speed': right_speed,
            'last_command_age': time.time() - self.last_command_time,
            'imu_status': imu_status
        }
    
    def _check_manual_mode(self) -> bool:
        if self.emergency_stopped:
            logger.warning("Cannot execute: Emergency stop active")
            return False
        if self.current_mode != 'manual':
            logger.warning(f"Cannot execute: Not in manual mode (current: {self.current_mode})")
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
                    logger.warning(f"Command timeout ({age:.1f}s) - Auto stopping")
                    self.stop()
            
            if self.current_state in ['MOVING FORWARD', 'MOVING BACKWARD', 
                                     'TURNING LEFT', 'TURNING RIGHT']:
                left, right = self.driver.get_speeds()
                if left == 0 and right == 0:
                    self.current_state = 'IDLE'

    def _vo_loop(self):
        """
        Visual Odometry background loop
        CHỈ chạy khi mode == 'manual' để tiết kiệm CPU
        """
        logger.info("🔄 VO background thread started")
        
        while self.running:
            try:
                if self.current_mode == 'manual':
                    # Lấy camera (lazy init)
                    if self.vo_camera is None:
                        try:
                            self.vo_camera = get_web_camera(self.config)
                            if not self.vo_camera.is_running():
                                self.vo_camera.start()
                        except Exception as e:
                            logger.error(f"❌ VO camera init error: {e}")
                            time.sleep(1.0)
                            continue
                    
                    # Capture frame
                    frame = self.vo_camera.capture_frame()
                    if frame is not None:
                        # Resize xuống 320x240 để VO chạy nhanh
                        frame_small = cv2.resize(frame, (320, 240))
                        
                        # Process frame (VO nhận BGR, tự convert sang grayscale)
                        self.vo.process_frame(frame_small)
                        
                        # Vẽ debug frame với features
                        self.vo_map = self.vo.draw_features(frame_small)
                    
                    time.sleep(0.1)  # ~10 FPS cho VO
                    
                else:
                    # Không phải manual mode -> ngủ dài để tiết kiệm CPU
                    self.vo_map = None  # Clear map khi không dùng
                    time.sleep(1.0)
                    
            except Exception as e:
                logger.error(f"❌ VO loop error: {e}")
                time.sleep(1.0)
        
        logger.info("🔄 VO background thread stopped")

    def get_vo_map(self):
        """Get current VO trajectory map (for display in manual mode)"""
        return self.vo_map

    def reset_odometry(self):
        """Reset visual odometry tracking"""
        if hasattr(self, 'vo') and self.vo is not None:
            self.vo.reset()
            self.vo_map = None
            logger.info("🔄 Visual Odometry reset")

    def cleanup(self):
        self.running = False
        if self.imu:
            self.imu.stop()
        self.driver.cleanup()
        logger.info("Robot Controller cleaned up")


# ============================================================
# AUTO MODE CONTROLLER (Lane Following)
# ============================================================

class AutoModeController:
    """Autonomous mode controller - Lane following with sign detection"""
    
    def __init__(self, robot_controller: RobotController):
        self.robot = robot_controller
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.camera: Optional[CameraManager] = None
        
        self.detector = ObjectDetector(
            model_path='data/models/best_ncnn_model', 
            conf_threshold=0.5
        )
        
        pid_config = robot_controller.config.get('lane_following', {}).get('pid', {})
        self.pid = PIDController(
            kp=pid_config.get('kp', 0.45),
            ki=pid_config.get('ki', 0.003),
            kd=pid_config.get('kd', 0.1),
            output_min=pid_config.get('min_output', -255),
            output_max=pid_config.get('max_output', 255),
            derivative_smoothing=pid_config.get('derivative_smoothing', 0.7)
        )
        
        lane_config = robot_controller.config.get('lane_following', {})
        self.base_speed = lane_config.get('base_speed', 110)
        self.default_speed = self.base_speed
        self.max_speed = lane_config.get('max_speed', 255)
        self.min_speed = lane_config.get('min_speed', 60)
        self.detection_config = robot_controller.config.get('ai', {}).get('lane_detection', {})
        
        # Sign detection thresholds
        self.DIST_PREPARE = 130
        self.DIST_EXECUTE = 300
        
        # Lane detection thresholds
        self.MAX_ERROR_THRESHOLD = 95
        self.lane_lost_count = 0
        self.lane_lost_threshold = 5
        
        # Lane Recovery System
        self.recovery_mode = False
        self.recovery_direction = 'left'
        self.recovery_scan_speed = 130
        self.recovery_scan_time = 0.0
        self.recovery_max_scan_time = 3.0
        self.recovery_attempts = 0
        self.recovery_max_attempts = 2
        
        # Smart Recovery: Lưu error cuối cùng khi còn thấy lane
        self.last_valid_error = 0.0
        
        # Low-Pass Filter (EMA) để làm mượt error
        self.filtered_error = 0.0
        self.smoothing_factor = 0.5  # Hệ số làm mượt (0.0-1.0)
        
        self.latest_debug_frame = None
        self.latest_error = 0
        self.latest_correction = 0
        
        logger.info("Auto Mode Controller initialized")
    
    def start(self):
        if not self.running:
            if not self._init_shared_camera():
                return False
            
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
                if not self.camera.start():
                    return False
            return True
        except Exception as e:
            logger.error(f"Camera init error: {e}")
            return False
    
    def _auto_loop(self):
        """Auto loop - Lane following with sign detection (optimized: no bounding box drawing)"""
        logger.info("Auto loop started")
        
        while self.running:
            try:
                if self.robot.current_mode != 'auto':
                    break
                
                frame = self.camera.capture_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                # Detect traffic signs (logic only, no drawing)
                detections, _ = self.detector.detect(frame)
                sign_action = None
                
                if detections:
                    sign = max(detections, key=lambda x: x['w'] * x['h'])
                    sign_name = sign['class_name']
                    sign_size = max(sign['w'], sign['h'])
                    
                    if sign_size < self.DIST_PREPARE:
                        self.robot.current_state = f"DETECTED: {sign_name} ({sign_size:.0f}px) - Too far"
                    
                    elif sign_size >= self.DIST_PREPARE and sign_size < self.DIST_EXECUTE:
                        self.robot.current_state = f"PREPARE: {sign_name} ({sign_size:.0f}px)"
                    
                    elif sign_size >= self.DIST_EXECUTE:
                        logger.info(f"🚦 EXECUTING: {sign_name} (Size: {sign_size:.0f}px)")
                        
                        if sign_name in ['stop_sign', 'red_light']:
                            self.robot.driver.stop()
                            sign_action = "STOP"
                            time.sleep(0.1)
                        
                        elif sign_name == 'green_light':
                            self.base_speed = self.default_speed
                        
                        elif sign_name == 'left_turn_sign':
                            logger.info("⬅️ Detected Left Turn Sign -> Smart Turn +90°")
                            self.robot.smart_turn(90, speed=220)
                            self.pid.reset()
                            continue
                        
                        elif sign_name == 'right_turn_sign':
                            logger.info("➡️ Detected Right Turn Sign -> Smart Turn -90°")
                            self.robot.smart_turn(-90, speed=220)
                            self.pid.reset()
                            continue
                        
                        elif sign_name == 'speed_limit_signs':
                            self.base_speed = 80
                        
                        elif sign_name == 'parking_signs':
                            self.robot.driver.stop()
                            self.stop()
                            break
                
                if sign_action in ["STOP", "TURN"]:
                    continue
                
                # Lane detection
                raw_error, x_line, center_x, lane_debug_frame = detect_line(
                    frame, self.detection_config
                )
                
                # Resize lane debug frame to 320x240 for reduced lag
                if lane_debug_frame is not None:
                    self.latest_debug_frame = cv2.resize(lane_debug_frame, (320, 240))
                else:
                    self.latest_debug_frame = None
                
                # Lane validity check (dùng raw_error để phản ứng nhanh khi mất lane)
                is_lane_valid = abs(raw_error) <= self.MAX_ERROR_THRESHOLD
                
                if not is_lane_valid:
                    self.lane_lost_count += 1
                    
                    logger.warning(f"⚠️ Lane lost! Error: {raw_error:.0f}px (Count: {self.lane_lost_count}/{self.lane_lost_threshold})")
                    
                    if self.lane_lost_count >= self.lane_lost_threshold:
                        if not self.recovery_mode:
                            # ===== SMART RECOVERY: Quyết định hướng quay dựa trên last_valid_error =====
                            if self.last_valid_error < 0:
                                # Làn đường nằm bên TRÁI → quay TRÁI trước
                                self.recovery_direction = 'left'
                                logger.info(f"🔍 SMART RECOVERY: Last error={self.last_valid_error:.0f}px (LEFT) → Scan LEFT first")
                            else:
                                # Làn đường nằm bên PHẢI → quay PHẢI trước
                                self.recovery_direction = 'right'
                                logger.info(f"🔍 SMART RECOVERY: Last error={self.last_valid_error:.0f}px (RIGHT) → Scan RIGHT first")
                            
                            self.recovery_mode = True
                            self.recovery_scan_time = 0.0
                            self.recovery_attempts = 0
                        
                        lane_found = self._perform_lane_recovery(frame)
                        
                        if lane_found:
                            logger.info("✅ Lane found! Resuming normal operation.")
                            self.recovery_mode = False
                            self.lane_lost_count = 0
                        elif self.recovery_attempts >= self.recovery_max_attempts:
                            logger.error("❌ Lane recovery failed! Robot STOPPED.")
                            self.robot.driver.stop()
                            self.robot.current_state = 'RECOVERY FAILED - STOPPED'
                            self.recovery_mode = False
                            time.sleep(1.0)
                        
                        continue
                    else:
                        self.robot.driver.stop()
                        self.robot.current_state = f'SEARCHING LANE ({self.lane_lost_count}/{self.lane_lost_threshold})'
                        time.sleep(0.05)
                        continue
                
                # Lane found - Cập nhật last_valid_error cho Smart Recovery
                self.last_valid_error = raw_error
                self.lane_lost_count = 0
                
                if self.recovery_mode:
                    logger.info("✅ Lane recovered during scan!")
                    self.recovery_mode = False
                    self.robot.driver.stop()
                    time.sleep(0.2)
                
                # Áp dụng Low-Pass Filter (EMA) để làm mượt error
                self.filtered_error = (self.smoothing_factor * raw_error) + ((1 - self.smoothing_factor) * self.filtered_error)
                
                # Cập nhật latest_error bằng giá trị đã lọc (hiển thị Dashboard mượt hơn)
                self.latest_error = int(self.filtered_error)
                
                if not detections:
                    self.robot.current_state = f'FOLLOWING LANE (Error: {self.latest_error:.0f}px)'
                
                # PID control (sử dụng filtered_error thay vì raw_error)
                current_time = time.time()
                dt = 0.05
                
                correction = self.pid.compute(self.filtered_error, dt)
                self.latest_correction = correction
                
                # Calculate motor speeds
                left_speed = max(-255, min(255, int(self.base_speed - correction)))
                right_speed = max(-255, min(255, int(self.base_speed + correction)))
                
                # Send to motors
                self.robot.driver.set_motors(left_speed, right_speed)
                
                time.sleep(0.03)
                
            except Exception as e:
                logger.error(f"❌ Error in auto loop: {e}")
                self.robot.driver.stop()
                break
        
        self.robot.driver.stop()
        logger.info("Auto loop ended")
    
    def _perform_lane_recovery(self, frame) -> bool:
        """Perform lane recovery by scanning left-right"""
        error, x_line, center_x, _ = detect_line(frame, self.detection_config)
        
        if abs(error) <= self.MAX_ERROR_THRESHOLD:
            return True
        
        self.recovery_scan_time += 0.05
        
        if self.recovery_scan_time >= self.recovery_max_scan_time:
            if self.recovery_direction == 'left':
                logger.info("🔄 Switching recovery scan direction: LEFT → RIGHT")
                self.recovery_direction = 'right'
            else:
                logger.info("🔄 Switching recovery scan direction: RIGHT → LEFT")
                self.recovery_direction = 'left'
                self.recovery_attempts += 1
            
            self.recovery_scan_time = 0.0
            
            if self.recovery_attempts >= self.recovery_max_attempts:
                return False
        
        if self.recovery_direction == 'left':
            self.robot.driver.turn_left(self.recovery_scan_speed)
            self.robot.current_state = f'SCANNING LEFT... ({self.recovery_scan_time:.1f}s)'
        else:
            self.robot.driver.turn_right(self.recovery_scan_speed)
            self.robot.current_state = f'SCANNING RIGHT... ({self.recovery_scan_time:.1f}s)'
        
        return False
    
    def get_debug_frame(self):
        return self.latest_debug_frame
    
    def get_pid_status(self):
        return {
            'error': self.latest_error,
            'correction': self.latest_correction,
            **self.pid.get_components()
        }


# ============================================================
# IMPROVED FOLLOW MODE CONTROLLER
# ============================================================

class FollowModeController:
    """
    IMPROVED Follow Mode Controller
    ✅ Target size = 200px (configurable)
    ✅ Forward if object < 200px (too far)
    ✅ Backward if object > 200px (too close)
    ✅ Left/Right centering with PID
    ✅ Support 4 colors: red, green, blue, yellow
    """
    
    def __init__(self, robot_controller: RobotController):
        self.robot = robot_controller
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.camera: Optional[CameraManager] = None
        
        # YOLO detector
        self.detector = ObjectDetector(
            model_path='data/models/best_ncnn_model', 
            conf_threshold=0.5
        )
        
        # ===== COLOR MAPPING =====
        self.color_map = {
            'red': 'red_color',
            'green': 'green_color',
            'blue': 'blue_color',
            'yellow': 'yellow_color'
        }
        self.target_color_name = 'red'
        
        # ===== TARGET SIZE CONTROL =====
        self.TARGET_SIZE = 350  # 🎯 Kích thước mục tiêu (pixels)
        self.SIZE_TOLERANCE = 20  # ±20px = dead zone (không điều chỉnh)
        
        # Size zones
        self.SIZE_MIN = self.TARGET_SIZE - self.SIZE_TOLERANCE  # 180px
        self.SIZE_MAX = self.TARGET_SIZE + self.SIZE_TOLERANCE  # 220px
        
        # ===== SPEED SETTINGS =====
        self.FORWARD_SPEED_MAX = 150   # Tốc độ tiến tối đa (khi object rất xa)
        self.FORWARD_SPEED_MIN = 80   # Tốc độ tiến tối thiểu (khi gần target)
        self.BACKWARD_SPEED = 100      # Tốc độ lùi (khi object quá gần)
        self.TURN_SPEED_MAX = 160      # Tốc độ quay tối đa (khi lệch nhiều)
        
        # ===== PID CONTROLLERS =====
        # PID cho điều khiển TRÁI/PHẢI (centering)
        self.pid_horizontal = PIDController(
            kp=0.3,   # Tăng để phản ứng nhanh hơn
            ki=0.0,
            kd=0.05,  # Giảm dao động
            output_min=-255,
            output_max=255
        )
        
        # PID cho điều khiển TIẾN/LÙI (distance control)
        self.pid_distance = PIDController(
            kp=1.0,   # Điều khiển khoảng cách
            ki=0.0,  # Xử lý sai số tích lũy
            kd=0.4,   # Giảm dao động
            output_min=-self.BACKWARD_SPEED,
            output_max=self.FORWARD_SPEED_MAX
        )
        
        # ===== TRACKING DATA =====
        self.target_x = 0
        self.target_y = 0
        self.target_w = 0
        self.target_h = 0
        self.confidence = 0
        self.target_distance = 0
        
        # Latest debug frame
        self.latest_debug_frame = None
        
        logger.info(f"✅ Improved Follow Mode initialized (Target: {self.TARGET_SIZE}px)")
    
    def start(self):
        if not self.running:
            if not self._init_shared_camera():
                return False
            
            self.pid_horizontal.reset()
            self.pid_distance.reset()
            
            self.running = True
            self.thread = threading.Thread(target=self._follow_loop, daemon=True)
            self.thread.start()
            
            logger.info(f"Follow mode started: {self.target_color_name}")
            return True
        return False
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        self.robot.driver.stop()
        logger.info("Follow mode stopped")
    
    def set_target_color(self, color: str):
        """Change target color"""
        if color in self.color_map:
            self.target_color_name = color
            logger.info(f"Target color changed to: {color}")
        else:
            logger.warning(f"Invalid color: {color}")
    
    def set_target_size(self, size: int):
        """
        Change target size dynamically
        Args:
            size: Target size in pixels (e.g., 150, 200, 250)
        """
        self.TARGET_SIZE = max(100, min(400, size))  # Clamp 100-400
        self.SIZE_MIN = self.TARGET_SIZE - self.SIZE_TOLERANCE
        self.SIZE_MAX = self.TARGET_SIZE + self.SIZE_TOLERANCE
        logger.info(f"Target size changed to: {self.TARGET_SIZE}px (±{self.SIZE_TOLERANCE}px)")
    
    def get_target_data(self) -> dict:
        """Get current tracking data"""
        return {
            'tracking': self.confidence > 0,
            'target_color': self.target_color_name,
            'target_x': self.target_x,
            'target_y': self.target_y,
            'target_w': self.target_w,
            'target_h': self.target_h,
            'confidence': self.confidence,
            'target_size': max(self.target_w, self.target_h) if self.target_w > 0 else 0,
            'target_size_desired': self.TARGET_SIZE
        }
    
    def get_debug_frame(self):
        """Get annotated debug frame"""
        return self.latest_debug_frame
    
    def _init_shared_camera(self) -> bool:
        try:
            self.camera = get_web_camera(self.robot.config)
            if not self.camera.is_running():
                if not self.camera.start():
                    return False
            return True
        except Exception as e:
            logger.error(f"Camera init error: {e}")
            return False

    def _follow_loop(self):
        """
        IMPROVED Follow Loop
        Sử dụng 2 PID controllers:
        - PID Horizontal: Điều chỉnh trái/phải (centering)
        - PID Distance: Điều chỉnh tiến/lùi (maintain target size)
        """
        logger.info(f"Follow loop started. Target: {self.color_map.get(self.target_color_name)}")
        
        while self.running:
            try:
                if self.robot.current_mode != 'follow':
                    break
                
                frame = self.camera.capture_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue
                
                # Detect objects
                detections, annotated_frame = self.detector.detect(frame)
                
                # Save debug frame
                self.latest_debug_frame = annotated_frame
                
                # Filter by target color
                target_class = self.color_map.get(self.target_color_name)
                valid_objs = [d for d in detections if d['class_name'] == target_class]
                
                if valid_objs:
                    # Get largest matching object
                    target = max(valid_objs, key=lambda x: x['w'] * x['h'])
                    
                    # ===== EXTRACT TARGET INFO =====
                    frame_h, frame_w = frame.shape[:2]
                    center_x = frame_w / 2
                    
                    self.target_x = int(target['x'])
                    self.target_y = int(target['y'])
                    self.target_w = int(target['w'])
                    self.target_h = int(target['h'])
                    self.confidence = int(target['conf'] * 100)
                    
                    # Object size (max dimension)
                    obj_size = max(self.target_w, self.target_h)
                    
                    # ===== PID 1: HORIZONTAL (Left/Right Centering) =====
                    # Error = target is on the LEFT → need to turn LEFT (negative error)
                    # Error = target is on the RIGHT → need to turn RIGHT (positive error)
                    error_horizontal = self.target_x - center_x
                    turn_correction = self.pid_horizontal.compute(error_horizontal)
                    
                    # ===== PID 2: DISTANCE (Forward/Backward) =====
                    # Error = object too small (far) → need to go FORWARD (negative error)
                    # Error = object too large (close) → need to go BACKWARD (positive error)
                    error_distance = obj_size - self.TARGET_SIZE
                    distance_correction = self.pid_distance.compute(error_distance)
                    
                    # ===== DETERMINE MOTION =====
                    
                    # 1. Check if within dead zone (no distance adjustment needed)
                    if self.SIZE_MIN <= obj_size <= self.SIZE_MAX:
                        # Perfect size - only center horizontally
                        base_speed = 0
                        status = f"LOCKED ON {target['class_name']} ({obj_size:.0f}px) ✓"
                    
                    elif obj_size < self.SIZE_MIN:
                        # Too small (too far) - move FORWARD
                        # Speed proportional to distance error
                        distance_error = self.TARGET_SIZE - obj_size
                        base_speed = int(self.FORWARD_SPEED_MIN + 
                                       (distance_error / self.TARGET_SIZE) * 
                                       (self.FORWARD_SPEED_MAX - self.FORWARD_SPEED_MIN))
                        base_speed = min(self.FORWARD_SPEED_MAX, base_speed)
                        status = f"APPROACHING {target['class_name']} ({obj_size:.0f}px) →"
                    
                    else:
                        # Too large (too close) - move BACKWARD
                        base_speed = -self.BACKWARD_SPEED
                        status = f"BACKING FROM {target['class_name']} ({obj_size:.0f}px) ←"
                    
                    # 2. Calculate final motor speeds
                    # Left motor: base_speed - turn_correction
                    # Right motor: base_speed + turn_correction
                    # (turn_correction < 0 → turn left, > 0 → turn right)
                    
                    left_speed = int(base_speed + turn_correction)
                    right_speed = int(base_speed - turn_correction)
                    
                    # Clamp to valid range
                    left_speed = max(-255, min(255, left_speed))
                    right_speed = max(-255, min(255, right_speed))
                    
                    # ===== SEND TO MOTORS =====
                    self.robot.driver.set_motors(left_speed, right_speed)
                    
                    # Update status
                    self.robot.current_state = status
                    
                    # ===== DRAW ENHANCED DEBUG INFO =====
                    if self.latest_debug_frame is not None:
                        h, w = self.latest_debug_frame.shape[:2]
                        
                        # Draw target size zone (green circle)
                        cv2.circle(self.latest_debug_frame, 
                                  (int(center_x), int(frame_h / 2)), 
                                  self.TARGET_SIZE, (0, 255, 0), 2)
                        cv2.putText(self.latest_debug_frame, 
                                   f"Target: {self.TARGET_SIZE}px", 
                                   (int(center_x) - self.TARGET_SIZE + 10, 
                                    int(frame_h / 2) - self.TARGET_SIZE - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
                        # Draw center line
                        cv2.line(self.latest_debug_frame, 
                                (int(center_x), 0), 
                                (int(center_x), h), 
                                (0, 255, 255), 1)
                        
                        # Draw object to center arrow
                        cv2.arrowedLine(self.latest_debug_frame,
                                       (int(center_x), h - 50),
                                       (self.target_x, h - 50),
                                       (255, 0, 255), 3, tipLength=0.3)
                        
                        # Draw info panel
                        info_y = 30
                        info_lines = [
                            f"Mode: FOLLOW",
                            f"Target: {self.target_color_name.upper()}",
                            f"Size: {obj_size:.0f}px (Goal: {self.TARGET_SIZE}px)",
                            f"H-Error: {error_horizontal:.0f}px",
                            f"D-Error: {error_distance:.0f}px",
                            f"L-Speed: {left_speed:+4d}",
                            f"R-Speed: {right_speed:+4d}"
                        ]
                        
                        for line in info_lines:
                            cv2.putText(self.latest_debug_frame, line,
                                       (10, info_y),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                            info_y += 20
                
                else:
                    # No target found - STOP and SEARCH
                    self.robot.driver.stop()
                    self.robot.current_state = f"SEARCHING {self.target_color_name.upper()}..."
                    self.confidence = 0
                    self.target_w = 0
                    self.target_h = 0
                    
                    # Draw search message
                    if self.latest_debug_frame is not None:
                        h, w = self.latest_debug_frame.shape[:2]
                        cv2.putText(self.latest_debug_frame,
                                   f"SEARCHING {self.target_color_name.upper()}...",
                                   (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                time.sleep(0.05)
                
            except Exception as e:
                logger.error(f"Follow loop error: {e}")
                self.robot.driver.stop()
                break
        
        self.robot.driver.stop()
        logger.info("Follow loop ended")