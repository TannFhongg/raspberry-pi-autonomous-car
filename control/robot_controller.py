"""
Robot Controller - COMPLETE VERSION
✅ Manual Mode
✅ Auto Mode (Lane Following)
✅ IMPROVED Follow Mode (Size-based distance control with PID)

CHANGELOG:
- FIX BUG-3: get_debug_frame() trả về .copy() thay vì reference
  → Flask thread đọc frame an toàn, không bị torn read
- FIX BUG-4: Không còn double-convert BGR (frame đã là BGR trước khi lưu vào buffer)
- FIX: Không cần sửa detect_line call vì đã fix signature ở lane_detector.py
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
        self.current_mode = 'auto'
        self.current_state = 'AUTO MODE'
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

        logger.info("Robot Controller initialized")
    
    def smart_turn(self, target_angle: float, speed: int = 200, timeout: float = 5.0):
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
        if mode in ['auto', 'follow']:
            self.current_mode = mode
            if mode == 'auto':
                self.current_state = 'AUTO MODE'
            elif mode == 'follow':
                self.current_state = 'FOLLOW MODE'
            logger.info(f"Mode changed to: {mode}")
            return True
        return False
    
    def set_speed(self, speed: int):
        self.current_speed = max(0, min(255, speed))
        logger.info(f"Speed set to: {self.current_speed}")
    
    def safe_set_motors(self, left_speed: int, right_speed: int) -> bool:
        if self.emergency_stopped:
            self.driver.stop()
            return False
        self.driver.set_motors(left_speed, right_speed)
        self._update_command_time()
        return True
    
    def safe_turn_left(self, speed: int) -> bool:
        if self.emergency_stopped:
            self.driver.stop()
            return False
        self.driver.turn_left(speed)
        self._update_command_time()
        return True
    
    def safe_turn_right(self, speed: int) -> bool:
        if self.emergency_stopped:
            self.driver.stop()
            return False
        self.driver.turn_right(speed)
        self._update_command_time()
        return True
    
    def stop(self):
        self.driver.stop()
        if self.current_mode == 'auto':
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
    
    def _update_command_time(self):
        self.last_command_time = time.time()
    
    def _watchdog(self):
        while self.running:
            time.sleep(0.5)

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
            model_path='models/best_ncnn_model', 
            conf_threshold=0.5
        )
        
        pid_config = robot_controller.config.get('lane_following', {}).get('pid', {})
        self.pid = PIDController(
            kp=pid_config.get('kp', 0.45),
            ki=pid_config.get('ki', 0.002),
            kd=pid_config.get('kd', 0.08),
            output_min=pid_config.get('min_output', -255),
            output_max=pid_config.get('max_output', 255),
            derivative_smoothing=pid_config.get('derivative_smoothing', 0.8)
        )
        
        lane_config = robot_controller.config.get('lane_following', {})
        self.base_speed = lane_config.get('base_speed', 120)
        self.default_speed = self.base_speed
        self.max_speed = lane_config.get('max_speed', 255)
        self.min_speed = lane_config.get('min_speed', 80)
        self.detection_config = robot_controller.config.get('ai', {}).get('lane_detection', {})
        
        self.DIST_PREPARE = 150
        self.DIST_EXECUTE = 250
        
        self.MAX_ERROR_THRESHOLD = 110
        self.lane_lost_count = 0
        self.lane_lost_threshold = 5
        
        self.approaching_turn_sign = False
        self.turn_sign_direction = None
        
        self.recovery_mode = False
        self.recovery_direction = 'left'
        self.recovery_scan_speed = 140
        self.recovery_start_time = 0.0
        self.recovery_max_scan_time = 0.5
        self.recovery_attempts = 0
        self.recovery_max_attempts = 2
        
        self.last_valid_error = 0.0
        
        self.filtered_error = 0.0
        self.smoothing_factor = 0.5
        
        self.last_time = None
        
        # ✅ FIX BUG-3: Lock và buffer cho debug frame
        self.debug_frame_lock = threading.Lock()
        self.latest_debug_frame = None
        
        logger.info("Auto Mode Controller initialized")
    
    def start(self):
        if not self.running:
            if not self._init_shared_camera():
                return False
            
            self.pid.reset()
            self.lane_lost_count = 0
            self.base_speed = self.default_speed
            self.last_time = None
            
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
        """Auto loop - Lane following with sign detection"""
        logger.info("Auto loop started")
        
        while self.running:
            try:
                if self.robot.current_mode != 'auto':
                    break
                
                frame_yuv = self.camera.capture_frame()
                if frame_yuv is None:
                    time.sleep(0.1)
                    continue
                
                # ============================================================
                # ✅ Convert YUV420→BGR một lần duy nhất
                # frame_bgr dùng cho: YOLO detection + debug frame storage
                # frame_yuv truyền vào lane_detector (lấy kênh Y trực tiếp)
                # ============================================================
                frame_bgr = cv2.cvtColor(frame_yuv, cv2.COLOR_YUV420p2BGR)
                
                # ✅ FIX BUG-3: Lưu COPY vào buffer, không lưu reference
                # Flask generator đọc từ buffer này → cần thread-safe copy
                frame_resized = cv2.resize(frame_bgr, (320, 240))
                with self.debug_frame_lock:
                    self.latest_debug_frame = frame_resized.copy()
                
                # Detect traffic signs
                detections, _ = self.detector.detect(frame_bgr)
                sign_action = None
                
                if detections:
                    sign = max(detections, key=lambda x: x['w'] * x['h'])
                    sign_name = sign['class_name']
                    sign_size = max(sign['w'], sign['h'])
                    
                    is_turn_sign = sign_name in ['left_turn_sign', 'right_turn_sign']
                    
                    if sign_size < self.DIST_PREPARE:
                        self.robot.current_state = f"DETECTED: {sign_name} ({sign_size:.0f}px) - Too far"
                        if is_turn_sign:
                            self.approaching_turn_sign = False
                            self.turn_sign_direction = None
                    
                    elif sign_size >= self.DIST_PREPARE and sign_size < self.DIST_EXECUTE:
                        self.robot.current_state = f"PREPARE: {sign_name} ({sign_size:.0f}px)"
                        if is_turn_sign:
                            self.approaching_turn_sign = True
                            self.turn_sign_direction = 'left' if sign_name == 'left_turn_sign' else 'right'
                            logger.info(f"🎯 APPROACHING {sign_name} - Will go STRAIGHT until close enough")
                    
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
                            self.approaching_turn_sign = False
                            self.turn_sign_direction = None
                            self.robot.smart_turn(80, speed=200)
                            self.pid.reset()
                            continue
                        
                        elif sign_name == 'right_turn_sign':
                            logger.info("➡️ Detected Right Turn Sign -> Smart Turn -90°")
                            self.approaching_turn_sign = False
                            self.turn_sign_direction = None
                            self.robot.smart_turn(-80, speed=200)
                            self.pid.reset()
                            continue
                        
                        elif sign_name == 'speed_limit_signs':
                            self.base_speed = 80
                        
                        elif sign_name == 'parking_signs':
                            self.robot.driver.stop()
                            self.stop()
                            break
                else:
                    if self.approaching_turn_sign:
                        logger.info("⚠️ Lost turn sign - Resuming normal lane following")
                        self.approaching_turn_sign = False
                        self.turn_sign_direction = None
                
                if sign_action in ["STOP", "TURN"]:
                    continue
                
                if self.approaching_turn_sign:
                    logger.info(f"🎯 APPROACHING TURN: Going STRAIGHT (Lane detection SKIPPED)")
                    self.robot.current_state = f'APPROACHING {self.turn_sign_direction.upper()} TURN (Straight)'
                    approach_speed = int(self.base_speed)
                    
                    if not self.robot.safe_set_motors(approach_speed, approach_speed):
                        logger.warning("⚠️ EMERGENCY STOP active - Approach blocked")
                        self.robot.current_state = 'EMERGENCY STOP'
                        time.sleep(0.1)
                        continue
                    
                    self.lane_lost_count = 0
                    time.sleep(0.03)
                    continue
                
                # ============================================================
                # ✅ Lane detection: Truyền RAW YUV420 (kênh Y = grayscale)
                # detect_line() giờ nhận debug=False (FIX BUG-1)
                # ============================================================
                raw_error, x_line, center_x, _ = detect_line(
                    frame_yuv, self.detection_config, debug=False
                )
                
                is_lane_valid = abs(raw_error) <= self.MAX_ERROR_THRESHOLD
                
                if not is_lane_valid:
                    self.lane_lost_count += 1
                    
                    logger.warning(f"⚠️ Lane lost! Error: {raw_error:.0f}px (Count: {self.lane_lost_count}/{self.lane_lost_threshold})")
                    
                    if self.lane_lost_count >= self.lane_lost_threshold:
                        if not self.recovery_mode:
                            if self.last_valid_error < 0:
                                self.recovery_direction = 'left'
                                logger.info(f"🔍 SMART RECOVERY: Last error={self.last_valid_error:.0f}px (LEFT) → Scan LEFT first")
                            else:
                                self.recovery_direction = 'right'
                                logger.info(f"🔍 SMART RECOVERY: Last error={self.last_valid_error:.0f}px (RIGHT) → Scan RIGHT first")
                            
                            self.recovery_mode = True
                            self.recovery_start_time = time.time()
                            self.recovery_attempts = 0
                        
                        lane_found = self._perform_lane_recovery(frame_yuv)
                        
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
                        
                        time.sleep(0.03)
                        continue
                    else:
                        self.robot.driver.stop()
                        self.robot.current_state = f'SEARCHING LANE ({self.lane_lost_count}/{self.lane_lost_threshold})'
                        time.sleep(0.03)
                        continue
                
                self.last_valid_error = raw_error
                self.lane_lost_count = 0
                
                if self.recovery_mode:
                    logger.info("✅ Lane recovered during scan!")
                    self.recovery_mode = False
                    self.robot.driver.stop()
                    time.sleep(0.2)
                
                self.filtered_error = (self.smoothing_factor * raw_error) + ((1 - self.smoothing_factor) * self.filtered_error)
                
                if not detections:
                    self.robot.current_state = f'FOLLOWING LANE (Error: {int(self.filtered_error):.0f}px)'
                
                current_time = time.time()
                if self.last_time is None:
                    dt = 0.03
                else:
                    dt = current_time - self.last_time
                    dt = max(0.01, min(0.2, dt))
                
                self.last_time = current_time
                
                correction = self.pid.compute(self.filtered_error, dt)
                
                left_speed = max(-255, min(255, int(self.base_speed - correction)))
                right_speed = max(-255, min(255, int(self.base_speed + correction)))
                
                if not self.robot.safe_set_motors(left_speed, right_speed):
                    logger.warning("⚠️ EMERGENCY STOP active - Auto mode blocked")
                    self.robot.current_state = 'EMERGENCY STOP'
                    time.sleep(0.1)
                    continue
                
                time.sleep(0.03)
                
            except Exception as e:
                logger.error(f"❌ Error in auto loop: {e}", exc_info=True)
                self.robot.driver.stop()
                # ✅ FIX: Không break ngay, thử recover sau 0.1s
                time.sleep(0.1)
                # Nếu lỗi liên tục sẽ thoát vòng lặp
                if not self.running:
                    break
        
        self.robot.driver.stop()
        logger.info("Auto loop ended")
    
    def _perform_lane_recovery(self, frame_yuv) -> bool:
        """Perform lane recovery by scanning left-right"""
        error, x_line, center_x, _ = detect_line(frame_yuv, self.detection_config, debug=False)
        
        if abs(error) <= self.MAX_ERROR_THRESHOLD:
            return True
        
        elapsed_time = time.time() - self.recovery_start_time
        
        if elapsed_time >= self.recovery_max_scan_time:
            if self.recovery_direction == 'left':
                logger.info("🔄 Switching recovery scan direction: LEFT → RIGHT")
                self.recovery_direction = 'right'
            else:
                logger.info("🔄 Switching recovery scan direction: RIGHT → LEFT")
                self.recovery_direction = 'left'
                self.recovery_attempts += 1
            
            self.recovery_start_time = time.time()
            
            if self.recovery_attempts >= self.recovery_max_attempts:
                return False
        
        if self.recovery_direction == 'left':
            if not self.robot.safe_turn_left(self.recovery_scan_speed):
                logger.warning("⚠️ EMERGENCY STOP active - Recovery blocked")
                return False
            self.robot.current_state = f'SCANNING LEFT... ({elapsed_time:.1f}s)'
        else:
            if not self.robot.safe_turn_right(self.recovery_scan_speed):
                logger.warning("⚠️ EMERGENCY STOP active - Recovery blocked")
                return False
            self.robot.current_state = f'SCANNING RIGHT... ({elapsed_time:.1f}s)'
        
        return False
    
    def get_debug_frame(self):
        """
        Get latest BGR frame for web dashboard (thread-safe)
        
        ✅ FIX BUG-3: Trả về COPY để Flask thread không bị torn read
        Buffer đã là BGR 320x240 - không cần convert thêm
        """
        with self.debug_frame_lock:
            if self.latest_debug_frame is None:
                return None
            # ✅ Trả về copy để tránh race condition với _auto_loop
            return self.latest_debug_frame.copy()
    
    def get_pid_status(self):
        return {
            'error': int(self.filtered_error),
            'correction': 0,
            **self.pid.get_components()
        }


# ============================================================
# IMPROVED FOLLOW MODE CONTROLLER
# ============================================================

class FollowModeController:
    """
    IMPROVED Follow Mode Controller
    ✅ Target size configurable from hardware_config.yaml
    """
    
    def __init__(self, robot_controller: RobotController):
        self.robot = robot_controller
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.camera: Optional[CameraManager] = None
        
        self.detector = ObjectDetector(
            model_path='models/best_ncnn_model', 
            conf_threshold=0.5
        )
        
        self.color_map = {
            'red': 'red_color',
            'green': 'green_color',
            'blue': 'blue_color',
            'yellow': 'yellow_color'
        }
        self.target_color_name = 'red'
        
        follow_config = robot_controller.config.get('follow_mode', {})
        
        self.TARGET_SIZE = follow_config.get('target_size', 350)
        self.SIZE_TOLERANCE = follow_config.get('size_tolerance', 20)
        self.SIZE_MIN = self.TARGET_SIZE - self.SIZE_TOLERANCE
        self.SIZE_MAX = self.TARGET_SIZE + self.SIZE_TOLERANCE
        
        self.FORWARD_SPEED_MAX = follow_config.get('forward_speed_max', 150)
        self.FORWARD_SPEED_MIN = follow_config.get('forward_speed_min', 80)
        self.BACKWARD_SPEED = follow_config.get('backward_speed', 100)
        self.TURN_SPEED_MAX = follow_config.get('turn_speed_max', 160)
        
        pid_h_config = follow_config.get('pid_horizontal', {})
        self.pid_horizontal = PIDController(
            kp=pid_h_config.get('kp', 0.3),
            ki=pid_h_config.get('ki', 0.0),
            kd=pid_h_config.get('kd', 0.05),
            output_min=-255,
            output_max=255
        )
        
        pid_d_config = follow_config.get('pid_distance', {})
        self.pid_distance = PIDController(
            kp=pid_d_config.get('kp', 1.0),
            ki=pid_d_config.get('ki', 0.0),
            kd=pid_d_config.get('kd', 0.4),
            output_min=-self.BACKWARD_SPEED,
            output_max=self.FORWARD_SPEED_MAX
        )
        
        self.target_x = 0
        self.target_y = 0
        self.target_w = 0
        self.target_h = 0
        self.confidence = 0
        self.target_distance = 0
        
        # ✅ FIX BUG-3: Lock cho latest_debug_frame
        self.debug_frame_lock = threading.Lock()
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
        if color in self.color_map:
            self.target_color_name = color
            logger.info(f"Target color changed to: {color}")
        else:
            logger.warning(f"Invalid color: {color}")
    
    def set_target_size(self, size: int):
        self.TARGET_SIZE = max(100, min(400, size))
        self.SIZE_MIN = self.TARGET_SIZE - self.SIZE_TOLERANCE
        self.SIZE_MAX = self.TARGET_SIZE + self.SIZE_TOLERANCE
        logger.info(f"Target size changed to: {self.TARGET_SIZE}px (±{self.SIZE_TOLERANCE}px)")
    
    def get_target_data(self) -> dict:
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
        """
        Get annotated debug frame (thread-safe)
        
        ✅ FIX BUG-3: Trả về COPY thay vì reference
        Buffer đã là BGR 320x240 với bounding boxes
        """
        with self.debug_frame_lock:
            if self.latest_debug_frame is None:
                return None
            # ✅ Trả về copy để Flask thread không bị corrupt
            return self.latest_debug_frame.copy()
    
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
        """IMPROVED Follow Loop"""
        logger.info(f"Follow loop started. Target: {self.color_map.get(self.target_color_name)}")
        
        while self.running:
            try:
                if self.robot.current_mode != 'follow':
                    break
                
                frame_yuv = self.camera.capture_frame()
                if frame_yuv is None:
                    time.sleep(0.05)
                    continue
                
                # ✅ Convert YUV420→BGR một lần duy nhất
                frame_bgr = cv2.cvtColor(frame_yuv, cv2.COLOR_YUV420p2BGR)
                
                # Detect objects với bounding boxes
                detections, annotated_frame = self.detector.detect(frame_bgr, draw_boxes=True)
                
                # ✅ FIX BUG-3: Lưu COPY vào buffer
                frame_resized = cv2.resize(annotated_frame, (320, 240))
                with self.debug_frame_lock:
                    self.latest_debug_frame = frame_resized.copy()
                
                target_class = self.color_map.get(self.target_color_name)
                valid_objs = [d for d in detections if d['class_name'] == target_class]
                
                if valid_objs:
                    target = max(valid_objs, key=lambda x: x['w'] * x['h'])
                    
                    frame_h, frame_w = frame_bgr.shape[:2]
                    center_x = frame_w / 2
                    
                    self.target_x = int(target['x'])
                    self.target_y = int(target['y'])
                    self.target_w = int(target['w'])
                    self.target_h = int(target['h'])
                    self.confidence = int(target['conf'] * 100)
                    
                    obj_size = max(self.target_w, self.target_h)
                    
                    error_horizontal = self.target_x - center_x
                    turn_correction = self.pid_horizontal.compute(error_horizontal)
                    
                    error_distance = obj_size - self.TARGET_SIZE
                    distance_correction = self.pid_distance.compute(error_distance)
                    
                    if self.SIZE_MIN <= obj_size <= self.SIZE_MAX:
                        base_speed = 0
                        status = f"LOCKED ON {target['class_name']} ({obj_size:.0f}px) ✓"
                    
                    elif obj_size < self.SIZE_MIN:
                        distance_error = self.TARGET_SIZE - obj_size
                        base_speed = int(self.FORWARD_SPEED_MIN + 
                                       (distance_error / self.TARGET_SIZE) * 
                                       (self.FORWARD_SPEED_MAX - self.FORWARD_SPEED_MIN))
                        base_speed = min(self.FORWARD_SPEED_MAX, base_speed)
                        status = f"APPROACHING {target['class_name']} ({obj_size:.0f}px) →"
                    
                    else:
                        base_speed = -self.BACKWARD_SPEED
                        status = f"BACKING FROM {target['class_name']} ({obj_size:.0f}px) ←"
                    
                    left_speed = int(base_speed + turn_correction)
                    right_speed = int(base_speed - turn_correction)
                    
                    left_speed = max(-255, min(255, left_speed))
                    right_speed = max(-255, min(255, right_speed))
                    
                    if not self.robot.safe_set_motors(left_speed, right_speed):
                        logger.warning("⚠️ EMERGENCY STOP active - Follow mode blocked")
                        self.robot.current_state = 'EMERGENCY STOP'
                        time.sleep(0.1)
                        continue
                    
                    self.robot.current_state = status
                
                else:
                    self.robot.driver.stop()
                    self.robot.current_state = f"SEARCHING {self.target_color_name.upper()}..."
                    self.confidence = 0
                    self.target_w = 0
                    self.target_h = 0
                
                time.sleep(0.05)
                
            except Exception as e:
                logger.error(f"Follow loop error: {e}", exc_info=True)
                self.robot.driver.stop()
                time.sleep(0.1)
                if not self.running:
                    break
        
        self.robot.driver.stop()
        logger.info("Follow loop ended")