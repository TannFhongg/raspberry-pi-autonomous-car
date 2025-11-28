"""
Robot Controller - FIXED VERSION
✅ Fixed typo: loger → logger
✅ Added IMU error handling
✅ Improved smart_turn with safety checks
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

logger = logging.getLogger(__name__)


class RobotController:
    """
    Main robot controller - FIXED VERSION
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

        # ===== IMU INITIALIZATION WITH ERROR HANDLING =====
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

        # Visual Odometry
        # Visual Odometry với scale calibration
        # Lưu ý: Cần chạy calibration tool để có scale chính xác
        # Giá trị 0.05 là ví dụ (1 pixel = 0.05 cm)
        vo_config = config.get('visual_odometry', {})
        scale = vo_config.get('scale_factor', 0.05)
        self.vo = VisualOdometry(scale_factor=scale)
        logger.info("✅ Visual Odometry initialized")


        logger.info("Robot Controller initialized")
    
    def smart_turn(self, target_angle: float, speed: int = 220, timeout: float = 5.0):
        """
        Rẽ chính xác sử dụng IMU (Closed-loop control)
        FIXED VERSION: Added error handling and safety checks
        
        Args:
            target_angle: Góc cần rẽ (Độ). 
                          +90 = Rẽ Trái 90 độ
                          -90 = Rẽ Phải 90 độ
            speed: Tốc độ động cơ khi rẽ (130-255)
            timeout: Thời gian tối đa (giây)
        """
        
        # ===== SAFETY CHECKS =====
        if speed < 130:
            logger.warning(f"Speed {speed} too low, setting to 130")
            speed = 130
        elif speed > 255:
            logger.warning(f"Speed {speed} too high, setting to 255")
            speed = 255
        
        if abs(target_angle) > 180:
            logger.error(f"❌ Invalid angle: {target_angle}° (must be -180 to 180)")
            return
        
        # ===== IMU CHECK =====
        if not hasattr(self, 'imu') or self.imu is None or not self.imu.connected:
            logger.warning("⚠️ IMU unavailable! Using time-based fallback.")
            self._fallback_turn(target_angle, speed)
            return
        
        # ===== SMART TURN WITH IMU =====
        logger.info(f"🔄 Smart Turn START: Target {target_angle}° at speed {speed}")
        
        try:
            # Reset góc về 0
            self.imu.reset_yaw()
            
            start_time = time.time()
            last_yaw = 0.0
            stuck_counter = 0  # Detect if robot is stuck
            
            while True:
                current_yaw = self.imu.get_yaw()
                error = abs(target_angle) - abs(current_yaw)
                
                # ===== STOP CONDITIONS =====
                
                # 1. Target reached
                if error <= 2.0:
                    logger.info(f"✅ Target Reached! Final: {current_yaw:.1f}°")
                    break
                
                # 2. Timeout
                if time.time() - start_time > timeout:
                    logger.warning(f"⚠️ Turn Timeout! Stopped at {current_yaw:.1f}° (Target: {target_angle}°)")
                    break
                
                # 3. Robot stuck detection (yaw not changing)
                if abs(current_yaw - last_yaw) < 0.1:
                    stuck_counter += 1
                    if stuck_counter > 50:  # Stuck for 0.5s
                        logger.error(f"❌ Robot appears stuck! Emergency stop.")
                        break
                else:
                    stuck_counter = 0
                
                last_yaw = current_yaw
                
                # 4. Overshoot protection
                if target_angle > 0 and current_yaw > target_angle + 5:
                    logger.warning(f"⚠️ Overshoot detected! {current_yaw:.1f}° > {target_angle}°")
                    break
                elif target_angle < 0 and current_yaw < target_angle - 5:
                    logger.warning(f"⚠️ Overshoot detected! {current_yaw:.1f}° < {target_angle}°")
                    break
                
                # ===== SPEED CONTROL (Adaptive) =====
                if error > 30:
                    current_speed = speed
                elif error > 10:
                    current_speed = int(speed * 0.7)
                else:
                    current_speed = max(130, int(speed * 0.5))
                
                # ===== MOTOR CONTROL =====
                if target_angle > 0:
                    self.driver.turn_left(current_speed)
                else:
                    self.driver.turn_right(current_speed)
                
                time.sleep(0.01)  # 100Hz update
            
        except Exception as e:
            logger.error(f"❌ Error during smart turn: {e}")
        
        finally:
            # Always stop motor
            self.driver.stop()
            time.sleep(0.2)
    
    def _fallback_turn(self, target_angle: float, speed: int):
        """
        Fallback: Rẽ theo thời gian khi IMU không khả dụng
        """
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
        
        # Include IMU status
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
# --------------------------------------------------------------    
    def update_odometry(self, frame):
        """
        Update visual odometry (gọi trong auto mode loop)

        Args:
            frame: Camera frame

        Returns:
            dict: Navigation status
        """
        if not hasattr(self, 'vo') or self.vo is None:
            return {'error': 'VO not initialized'}

        try:
            # Process frame
            dx, dy = self.vo.process_frame(frame)

            # Get comprehensive status
            status = self.vo.get_status()
            status['dx'] = dx
            status['dy'] = dy

            return status

        except Exception as e:
            logger.error(f"❌ VO update error: {e}")
            return {'error': str(e)}

    def reset_odometry(self):
        """Reset visual odometry tracking"""
        if hasattr(self, 'vo') and self.vo is not None:
            self.vo.reset()
            logger.info("🔄 Visual Odometry reset")
# --------------------------------------------------------------

    def cleanup(self):
        self.running = False
        
        # Stop IMU thread
        if self.imu:
            self.imu.stop()
        
        self.driver.cleanup()
        logger.info("Robot Controller cleaned up")


# ===== AUTO MODE CONTROLLER (With Fixed Logger) =====
class AutoModeController:
    """
    Autonomous mode controller - FIXED VERSION
    """
    
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
            kp=pid_config.get('kp', 0.8),
            ki=pid_config.get('ki', 0.0),
            kd=pid_config.get('kd', 0.3),
            output_min=pid_config.get('min_output', -255),
            output_max=pid_config.get('max_output', 255),
            derivative_smoothing=pid_config.get('derivative_smoothing', 0.7)
        )
        
        lane_config = robot_controller.config.get('lane_following', {})
        self.base_speed = lane_config.get('base_speed', 150)
        self.default_speed = self.base_speed
        self.max_speed = lane_config.get('max_speed', 255)
        self.min_speed = lane_config.get('min_speed', 60)
        self.detection_config = robot_controller.config.get('ai', {}).get('lane_detection', {})
        
        self.DIST_PREPARE = 140
        self.DIST_EXECUTE = 170
        
        self.lane_lost_count = 0
        self.lane_lost_threshold = 10
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
        logger.info("Auto loop started")
        
        while self.running:
            try:
                if self.robot.current_mode != 'auto':
                    break
                
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
                    
                    if sign_size < self.DIST_PREPARE:
                        pass
                    elif sign_size > self.DIST_EXECUTE + 20:
                        pass
                    else:
                        logger.info(f"🚦 EXECUTING: {sign_name} (Size: {sign_size:.0f})")
                        
                        if sign_name in ['stop_sign', 'red_light']:
                            self.robot.driver.stop()
                            sign_action = "STOP"
                            time.sleep(0.1)
                        
                        elif sign_name == 'green_light':
                            self.base_speed = self.default_speed
                        
                        elif sign_name == 'left_turn_sign':
                            # ✅ FIXED: logger (not loger)
                            logger.info("⬅️ Detected Left Turn Sign -> Smart Turn +90°")
                            self.robot.smart_turn(90, speed=220)
                            continue
                        
                        elif sign_name == 'right_turn_sign':
                            # ✅ FIXED: logger (not loger)
                            logger.info("➡️ Detected Right Turn Sign -> Smart Turn -90°")
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
                error, x_line, center_x, lane_debug_frame = detect_line(
                    frame, self.detection_config
                )
                self.latest_debug_frame = lane_debug_frame
                self.latest_error = error
                
                current_time = time.time()
                dt = 0.05
                
                if abs(error) > frame.shape[1] * 0.4:
                    self.lane_lost_count += 1
                    if self.lane_lost_count >= self.lane_lost_threshold:
                        self.robot.driver.stop()
                        self.robot.current_state = 'LANE LOST'
                        continue
                else:
                    self.lane_lost_count = 0
                    if not detections:
                        self.robot.current_state = 'FOLLOWING LANE'
                
                correction = self.pid.compute(error, dt)
                self.latest_correction = correction
                
                left_speed = max(-255, min(255, int(self.base_speed - correction)))
                right_speed = max(-255, min(255, int(self.base_speed + correction)))
                
                self.robot.driver.set_motors(left_speed, right_speed)
                time.sleep(0.03)
                
            except Exception as e:
                logger.error(f"❌ Error in auto loop: {e}")
                self.robot.driver.stop()
                break
        
        self.robot.driver.stop()
        logger.info("Auto loop ended")
    
    def get_debug_frame(self):
        return self.latest_debug_frame
    
    def get_pid_status(self):
        return {
            'error': self.latest_error,
            'correction': self.latest_correction,
            **self.pid.get_components()
        }