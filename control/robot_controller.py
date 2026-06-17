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
from perception.camera_manager import CameraManager, frame_to_bgr, get_web_camera
from perception.object_detector import ObjectDetector
from perception.imu_sensor_fusion import IMUSensorFusion

import cv2

logger = logging.getLogger(__name__)


def _create_dashboard_frame(frame, size=(320, 240), format_name=None):
    """
    Create a clean BGR frame for the web dashboard.

    The capture format can be YUV420, RGB888, BGR888, or grayscale depending
    on hardware_config.yaml. Convert it once before sending to the dashboard.
    """
    if frame is None:
        return None

    frame_bgr = frame_to_bgr(frame, format_name)
    if frame_bgr is None or len(frame_bgr.shape) != 3:
        logger.warning(f"Unexpected dashboard frame format: shape={frame.shape}")
        return None

    return cv2.resize(frame_bgr, size, interpolation=cv2.INTER_AREA)


def _store_latest_debug_frame(lock, owner, frame):
    camera = getattr(owner, "camera", None)
    dashboard_frame = _create_dashboard_frame(
        frame,
        format_name=getattr(camera, "format", None),
    )
    if dashboard_frame is None:
        return

    with lock:
        owner.latest_debug_frame = dashboard_frame


# ============================================================
# ROBOT CONTROLLER (Base Class)
# ============================================================

class RobotController:
    """Main robot controller"""

    def __init__(self, motor_driver, config: dict):
        self.driver = motor_driver
        self.config = config

        # Current state
        self.current_mode = 'idle'
        self.current_state = 'STANDBY'
        self.current_speed = self._clamp_speed(
            config.get('robot', {}).get('default_speed', 180)
        )

        # Safety
        self.emergency_stopped = False
        self.last_command_time = time.time()
        self.timeout = config.get('safety', {}).get('timeout', 5.0)
        self.watchdog_triggered = False

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
        if self.emergency_stopped:
            logger.warning("Smart turn blocked: emergency stop is active")
            self.driver.stop()
            return False

        if speed < 130:
            logger.warning(f"Speed {speed} too low, setting to 130")
            speed = 130
        elif speed > 255:
            logger.warning(f"Speed {speed} too high, setting to 255")
            speed = 255

        if abs(target_angle) > 180:
            logger.error(f"❌ Invalid angle: {target_angle}° (must be -180 to 180)")
            return False

        if not hasattr(self, 'imu') or self.imu is None or not self.imu.connected:
            logger.warning("⚠️ IMU unavailable! Using time-based fallback.")
            return self._fallback_turn(target_angle, speed)

        logger.info(f"🔄 Smart Turn START: Target {target_angle}° at speed {speed}")

        try:
            self.imu.reset_yaw()
            start_time = time.time()
            last_yaw = 0.0
            stuck_counter = 0

            while True:
                if self.emergency_stopped:
                    logger.warning("Smart turn interrupted by emergency stop")
                    return False

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
                    if not self.safe_turn_left(current_speed):
                        return False
                else:
                    if not self.safe_turn_right(current_speed):
                        return False

                time.sleep(0.01)

        except Exception as e:
            logger.error(f"❌ Error during smart turn: {e}")
            return False

        finally:
            self.driver.stop()
            time.sleep(0.2)

        return True

    def _fallback_turn(self, target_angle: float, speed: int):
        """Fallback turn based on time"""
        if self.emergency_stopped:
            logger.warning("Fallback turn blocked: emergency stop is active")
            self.driver.stop()
            return False

        duration = 0.6 * (abs(target_angle) / 90.0)
        logger.info(f"⏱️ Fallback Turn: {target_angle}° for {duration:.2f}s")

        start_time = time.time()
        while time.time() - start_time < duration:
            if self.emergency_stopped:
                logger.warning("Fallback turn interrupted by emergency stop")
                self.driver.stop()
                return False

            if target_angle > 0:
                if not self.safe_turn_left(speed):
                    return False
            else:
                if not self.safe_turn_right(speed):
                    return False

            time.sleep(0.02)

        self.driver.stop()
        return True

    def set_mode(self, mode: str):
        if mode in ['auto', 'follow', 'idle']:
            self.current_mode = mode
            if mode == 'auto':
                self.current_state = 'AUTO MODE'
            elif mode == 'follow':
                self.current_state = 'FOLLOW MODE'
            elif mode == 'idle':
                self.current_state = 'STANDBY'
            logger.info(f"Mode changed to: {mode}")
            return True
        return False

    @staticmethod
    def _clamp_speed(speed: int) -> int:
        return max(0, min(255, int(speed)))

    def _limit_motor_speed(self, speed: int) -> int:
        speed = max(-255, min(255, int(speed)))
        if speed > 0:
            return min(speed, self.current_speed)
        if speed < 0:
            return max(speed, -self.current_speed)
        return 0

    def set_speed(self, speed: int) -> int:
        self.current_speed = self._clamp_speed(speed)
        self.watchdog_triggered = False
        logger.info(f"Speed set to: {self.current_speed}")
        return self.current_speed

    def safe_set_motors(self, left_speed: int, right_speed: int) -> bool:
        """
        Safe motor control with emergency stop check

        ✅ CRITICAL SAFETY: Chặn lệnh nếu emergency_stopped = True

        Args:
            left_speed: Left motor speed (-255 to 255)
            right_speed: Right motor speed (-255 to 255)

        Returns:
            True if command executed, False if blocked by emergency stop
        """
        if self.emergency_stopped:
            # ⚠️ EMERGENCY STOP ACTIVE - Block all motor commands
            self.driver.stop()
            return False

        left_speed = self._limit_motor_speed(left_speed)
        right_speed = self._limit_motor_speed(right_speed)

        # Normal operation - execute command
        self.driver.set_motors(left_speed, right_speed)
        self._update_command_time()
        return True

    def safe_turn_left(self, speed: int) -> bool:
        """Safe turn left with emergency stop check"""
        if self.emergency_stopped:
            self.driver.stop()
            return False
        speed = self._clamp_speed(speed)
        speed = min(speed, self.current_speed)
        self.driver.turn_left(speed)
        self._update_command_time()
        return True

    def safe_turn_right(self, speed: int) -> bool:
        """Safe turn right with emergency stop check"""
        if self.emergency_stopped:
            self.driver.stop()
            return False
        speed = self._clamp_speed(speed)
        speed = min(speed, self.current_speed)
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
        self.watchdog_triggered = False

    def check_watchdog_timeout(self, now: Optional[float] = None) -> bool:
        """Stop motors if active command heartbeat has expired."""
        if self.emergency_stopped or self.timeout <= 0:
            return False

        now = time.time() if now is None else now
        if now - self.last_command_time <= self.timeout:
            self.watchdog_triggered = False
            return False

        try:
            left_speed, right_speed = self.driver.get_speeds()
        except Exception as e:
            logger.error(f"Watchdog could not read motor speeds: {e}")
            left_speed, right_speed = 0, 0

        active_mode = self.current_mode in ['auto', 'follow']
        motors_active = left_speed != 0 or right_speed != 0

        if not active_mode and not motors_active:
            return False

        self.driver.stop()
        self.current_mode = 'idle'
        self.current_state = 'WATCHDOG TIMEOUT - STOPPED'

        if not self.watchdog_triggered:
            age = now - self.last_command_time
            logger.warning(
                f"Watchdog timeout: no valid command for {age:.1f}s "
                f"(limit {self.timeout:.1f}s). Motors stopped."
            )

        self.watchdog_triggered = True
        return True

    def _watchdog(self):
        while self.running:
            time.sleep(0.5)
            self.check_watchdog_timeout()

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

        # PID config - Giá trị mặc định khớp với hardware_config.yaml
        pid_config = robot_controller.config.get('lane_following', {}).get('pid', {})
        self.pid = PIDController(
            kp=pid_config.get('kp', 0.45),              # Khớp với hardware_config.yaml
            ki=pid_config.get('ki', 0.002),              # Khớp với hardware_config.yaml
            kd=pid_config.get('kd', 0.08),             # Khớp với hardware_config.yaml
            output_min=pid_config.get('min_output', -255),
            output_max=pid_config.get('max_output', 255),
            derivative_smoothing=pid_config.get('derivative_smoothing', 0.8)
        )

        # Speed config - Giá trị mặc định khớp với hardware_config.yaml
        lane_config = robot_controller.config.get('lane_following', {})
        self.base_speed = lane_config.get('base_speed', 120)    # Khớp với hardware_config.yaml
        self.default_speed = self.base_speed
        self.max_speed = lane_config.get('max_speed', 255)     # Khớp với hardware_config.yaml
        self.min_speed = lane_config.get('min_speed', 80)      # Khớp với hardware_config.yaml
        self.turn_sign_angle = float(lane_config.get('turn_sign_angle', 72))
        self.turn_sign_speed = int(lane_config.get('turn_sign_speed', 170))
        self.detection_config = robot_controller.config.get('ai', {}).get('lane_detection', {})

        # Sign detection thresholds - đọc từ hardware_config.yaml để dễ calibrate
        sign_config = lane_config.get('sign_detection', {}) or {}
        self.DIST_PREPARE = int(
            sign_config.get('dist_prepare', lane_config.get('dist_prepare', 150))
        )
        self.DIST_EXECUTE = int(
            sign_config.get('dist_execute', lane_config.get('dist_execute', 250))
        )
        if self.DIST_EXECUTE <= self.DIST_PREPARE:
            logger.warning(
                "Invalid sign detection thresholds: dist_execute=%s <= dist_prepare=%s. "
                "Using dist_execute=dist_prepare+1.",
                self.DIST_EXECUTE,
                self.DIST_PREPARE,
            )
            self.DIST_EXECUTE = self.DIST_PREPARE + 1

        # Lane detection thresholds
        self.MAX_ERROR_THRESHOLD = 110  # Sai số tối đa để coi là còn lane
        self.lane_lost_count = 0
        self.lane_lost_threshold = 5

        # ===== TURN SIGN APPROACH MODE =====
        # Khi phát hiện biển rẽ ở giai đoạn PREPARE, xe sẽ đi thẳng cho đến khi EXECUTE
        self.approaching_turn_sign = False
        self.turn_sign_direction = None  # 'left' hoặc 'right'

        # Lane Recovery System
        self.recovery_mode = False
        self.recovery_direction = 'left'
        self.recovery_scan_speed = 140
        self.recovery_start_time = 0.0  # ✅ FIX: Wall clock time thay vì counter
        self.recovery_max_scan_time = 0.5  # Giây để quét mỗi bên
        self.recovery_attempts = 0
        self.recovery_max_attempts = 2

        # Smart Recovery: Lưu error cuối cùng khi còn thấy lane
        self.last_valid_error = 0.0

        # Low-Pass Filter (EMA) để làm mượt error
        self.filtered_error = 0.0
        self.smoothing_factor = 0.5  # Hệ số làm mượt (0.0-1.0)

        # ===== TIMING FIX: Dynamic dt calculation =====
        self.last_time = None  # Track thời gian thực tế giữa các iteration

        # ===== WEB DASHBOARD: Lưu frame BGR để hiển thị =====
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

            # ===== TIMING FIX: Reset timing khi start =====
            self.last_time = None  # Reset để tính dt từ đầu

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

    def set_speed(self, speed: int) -> int:
        """Update auto lane-following base speed at runtime."""
        speed = self.robot._clamp_speed(speed)
        self.default_speed = speed
        self.base_speed = speed
        logger.info(f"Auto mode speed updated: base_speed={self.base_speed}")
        return self.base_speed

    def _init_shared_camera(self) -> bool:
        try:
            self.camera = get_web_camera(self.robot.config)
            if not self.camera.is_running():
                if not self.camera.start():
                    logger.error("Camera failed to start for auto mode")
                    return False
            return True
        except Exception as e:
            logger.error(f"Camera init error for auto mode: {e}")
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
                    time.sleep(0.005)
                    continue

                _store_latest_debug_frame(self.debug_frame_lock, self, frame)

                frame_bgr = frame_to_bgr(frame, self.camera.format)
                lane_frame = frame if str(self.camera.format).upper() == 'YUV420' else frame_bgr

                # Detect traffic signs (logic only, no drawing)
                detections, _ = self.detector.detect(frame_bgr)
                sign_action = None

                if detections:
                    sign = max(detections, key=lambda x: x['w'] * x['h'])
                    sign_name = sign['class_name']
                    sign_size = max(sign['w'], sign['h'])

                    # ===== TURN SIGN APPROACH LOGIC =====
                    # Khi phát hiện biển rẽ, đánh dấu để xử lý đặc biệt
                    is_turn_sign = sign_name in ['left_turn_sign', 'right_turn_sign']

                    if sign_size < self.DIST_PREPARE:
                        self.robot.current_state = f"DETECTED: {sign_name} ({sign_size:.0f}px) - Too far"
                        # Reset approach mode nếu biển quá xa
                        if is_turn_sign:
                            self.approaching_turn_sign = False
                            self.turn_sign_direction = None

                    elif sign_size >= self.DIST_PREPARE and sign_size < self.DIST_EXECUTE:
                        self.robot.current_state = f"PREPARE: {sign_name} ({sign_size:.0f}px)"

                        # ===== QUAN TRỌNG: Khi tiếp cận biển rẽ, bật chế độ đi thẳng =====
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
                            # Reset approach mode trước khi rẽ
                            self.approaching_turn_sign = False
                            self.turn_sign_direction = None
                            self.robot.smart_turn(self.turn_sign_angle, speed=self.turn_sign_speed)
                            self.last_time = None # ✅ THÊM DÒNG NÀY (Reset rác thời gian)
                            self.pid.reset()
                            continue

                        elif sign_name == 'right_turn_sign':
                            logger.info("➡️ Detected Right Turn Sign -> Smart Turn -90°")
                            # Reset approach mode trước khi rẽ
                            self.approaching_turn_sign = False
                            self.turn_sign_direction = None
                            self.robot.smart_turn(-self.turn_sign_angle, speed=self.turn_sign_speed)
                            self.last_time = None # ✅ THÊM DÒNG NÀY (Reset rác thời gian)
                            self.pid.reset()
                            continue

                        elif sign_name == 'speed_limit_signs':
                            self.base_speed = 80

                        elif sign_name == 'parking_signs':
                            self.robot.driver.stop()
                            self.stop()
                            break
                else:
                    # Không thấy biển nào → reset approach mode
                    if self.approaching_turn_sign:
                        logger.info("⚠️ Lost turn sign - Resuming normal lane following")
                        self.approaching_turn_sign = False
                        self.turn_sign_direction = None

                if sign_action in ["STOP", "TURN"]:
                    continue

                # ===== TURN SIGN APPROACH: Đi thẳng, BỎ QUA lane detection =====
                if self.approaching_turn_sign:
                    logger.info(f"🎯 APPROACHING TURN: Going STRAIGHT (Lane detection SKIPPED)")
                    self.robot.current_state = f'APPROACHING {self.turn_sign_direction.upper()} TURN (Straight)'

                    # Đi thẳng với tốc độ base
                    approach_speed = int(self.base_speed)

                    # ✅ SAFETY: Dùng safe_set_motors thay vì gọi trực tiếp driver
                    if not self.robot.safe_set_motors(approach_speed, approach_speed):
                        logger.warning("⚠️ EMERGENCY STOP active - Approach blocked")
                        self.robot.current_state = 'EMERGENCY STOP'
                        time.sleep(0.1)
                        continue

                    # Reset lane lost count
                    self.lane_lost_count = 0
                    time.sleep(0.03)
                    continue

                # ============================================================
                # Lane detection (CHỈ chạy khi KHÔNG approaching turn sign)
                # ✅ OPTIMIZED: Truyền RAW YUV420 vào lane_detector
                # → lane_detector sẽ lấy trực tiếp kênh Y (CPU cost = 0)
                # ✅ PERFORMANCE: debug=False để tắt vẽ hình (tăng hiệu suất)
                # ============================================================
                raw_error, x_line, center_x, _ = detect_line(
                    lane_frame, self.detection_config, debug=False
                )

                # Lane validity check
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
                            self.recovery_start_time = time.time()  # ✅ FIX: Dùng wall clock
                            self.recovery_attempts = 0

                        lane_found = self._perform_lane_recovery(lane_frame)

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

                        # ✅ FIX: Vẫn cập nhật debug frame khi recovery để tránh đứng hình
                        time.sleep(0.03)
                        continue
                    else:
                        self.robot.driver.stop()
                        self.robot.current_state = f'SEARCHING LANE ({self.lane_lost_count}/{self.lane_lost_threshold})'
                        # ✅ FIX: Bỏ sleep dài, giữ frame rate ổn định
                        time.sleep(0.03)
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

                if not detections:
                    self.robot.current_state = f'FOLLOWING LANE (Error: {int(self.filtered_error):.0f}px)'

                # ===== PID CONTROL với DYNAMIC dt =====
                current_time = time.time()

                # Tính dt thực tế từ iteration trước
                if self.last_time is None:
                    # Iteration đầu tiên: dùng giá trị mặc định
                    dt = 0.03  # Khớp với time.sleep(0.03) ở cuối loop
                else:
                    dt = current_time - self.last_time
                    # Safety check: tránh dt quá nhỏ hoặc quá lớn
                    dt = max(0.01, min(0.2, dt))  # Clamp dt trong khoảng 10ms-200ms

                self.last_time = current_time

                correction = self.pid.compute(self.filtered_error, dt)

                # Calculate motor speeds
                left_speed = max(-255, min(255, int(self.base_speed - correction)))
                right_speed = max(-255, min(255, int(self.base_speed + correction)))

                # Send to motors
                # ✅ SAFETY: Dùng safe_set_motors thay vì gọi trực tiếp driver
                if not self.robot.safe_set_motors(left_speed, right_speed):
                    logger.warning("⚠️ EMERGENCY STOP active - Auto mode blocked")
                    self.robot.current_state = 'EMERGENCY STOP'
                    time.sleep(0.1)
                    continue

                time.sleep(0.03)

            except Exception as e:
                logger.error(f"❌ Error in auto loop: {e}")
                self.robot.driver.stop()
                break

        self.robot.driver.stop()
        logger.info("Auto loop ended")

    def _perform_lane_recovery(self, frame_yuv) -> bool:
        """
        Perform lane recovery by scanning left-right

        ✅ FIX: Dùng wall clock time thay vì fake counter
        ✅ OPTIMIZED: Nhận RAW YUV420, truyền trực tiếp vào lane_detector
        """
        error, x_line, center_x, _ = detect_line(frame_yuv, self.detection_config, debug=False)

        if abs(error) <= self.MAX_ERROR_THRESHOLD:
            return True

        # ✅ FIX: Tính elapsed time từ wall clock
        elapsed_time = time.time() - self.recovery_start_time

        if elapsed_time >= self.recovery_max_scan_time:
            if self.recovery_direction == 'left':
                logger.info("🔄 Switching recovery scan direction: LEFT → RIGHT")
                self.recovery_direction = 'right'
            else:
                logger.info("🔄 Switching recovery scan direction: RIGHT → LEFT")
                self.recovery_direction = 'left'
                self.recovery_attempts += 1

            # Reset timer cho direction mới
            self.recovery_start_time = time.time()

            if self.recovery_attempts >= self.recovery_max_attempts:
                return False

        if self.recovery_direction == 'left':
            # ✅ SAFETY: Dùng safe_turn_left thay vì gọi trực tiếp driver
            if not self.robot.safe_turn_left(self.recovery_scan_speed):
                logger.warning("⚠️ EMERGENCY STOP active - Recovery blocked")
                return False
            self.robot.current_state = f'SCANNING LEFT... ({elapsed_time:.1f}s)'
        else:
            # ✅ SAFETY: Dùng safe_turn_right thay vì gọi trực tiếp driver
            if not self.robot.safe_turn_right(self.recovery_scan_speed):
                logger.warning("⚠️ EMERGENCY STOP active - Recovery blocked")
                return False
            self.robot.current_state = f'SCANNING RIGHT... ({elapsed_time:.1f}s)'

        return False

    def get_debug_frame(self):
        """
        Get latest BGR frame for web dashboard (thread-safe)
        """
        with self.debug_frame_lock:
            if self.latest_debug_frame is None:
                return None
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
    ✅ Forward if object < target_size (too far)
    ✅ Backward if object > target_size (too close)
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
            model_path='models/best_ncnn_model',
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

        # ===== LOAD CONFIG FROM hardware_config.yaml =====
        follow_config = robot_controller.config.get('follow_mode', {})

        # Target size control - Đọc từ config
        self.TARGET_SIZE = follow_config.get('target_size', 350)
        self.SIZE_TOLERANCE = follow_config.get('size_tolerance', 20)

        # Size zones
        self.SIZE_MIN = self.TARGET_SIZE - self.SIZE_TOLERANCE  # 180px
        self.SIZE_MAX = self.TARGET_SIZE + self.SIZE_TOLERANCE  # 220px

        # ===== SPEED SETTINGS - Đọc từ config =====
        self.FORWARD_SPEED_MAX = follow_config.get('forward_speed_max', 150)
        self.FORWARD_SPEED_MIN = follow_config.get('forward_speed_min', 80)
        self.BACKWARD_SPEED_MIN = follow_config.get('backward_speed_min', 80)
        self.BACKWARD_SPEED = follow_config.get('backward_speed', 100)
        self.BACKING_STRAIGHT_MARGIN = follow_config.get('backing_straight_margin', 60)
        self.BACKING_TURN_RATIO = follow_config.get('backing_turn_ratio', 0.4)
        self.TURN_SPEED_MAX = follow_config.get('turn_speed_max', 160)
        self.TURN_SPEED_MIN = follow_config.get('turn_speed_min', 0)
        self.HORIZONTAL_TOLERANCE = follow_config.get('horizontal_tolerance', 35)
        self.speed_limit = robot_controller.current_speed
        self.last_time = None  # Track real time between PID updates

        # ===== PID CONTROLLERS - Đọc từ config =====
        # PID cho điều khiển TRÁI/PHẢI (centering)
        pid_h_config = follow_config.get('pid_horizontal', {})
        self.pid_horizontal = PIDController(
            kp=pid_h_config.get('kp', 0.3),
            ki=pid_h_config.get('ki', 0.0),
            kd=pid_h_config.get('kd', 0.05),
            output_min=-255,
            output_max=255
        )

        # PID cho điều khiển TIẾN/LÙI (distance control)
        pid_d_config = follow_config.get('pid_distance', {})
        self.pid_distance = PIDController(
            kp=pid_d_config.get('kp', 1.0),
            ki=pid_d_config.get('ki', 0.0),
            kd=pid_d_config.get('kd', 0.4),
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

        # ===== FIX RACE CONDITION: Lock cho latest_debug_frame =====
        # Flask /debug_feed đọc frame này → cần lock để tránh corrupt
        self.debug_frame_lock = threading.Lock()
        self.latest_debug_frame = None

        logger.info(f"✅ Improved Follow Mode initialized (Target: {self.TARGET_SIZE}px)")

    def start(self):
        if not self.running:
            if not self._init_shared_camera():
                return False

            self.pid_horizontal.reset()
            self.pid_distance.reset()
            self.last_time = None

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

    def set_speed(self, speed: int) -> int:
        """Update follow mode runtime speed cap."""
        self.speed_limit = self.robot._clamp_speed(speed)
        logger.info(f"Follow mode speed limit updated: {self.speed_limit}")
        return self.speed_limit

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
        """
        Get clean dashboard frame (thread-safe)

        ✅ FIX: Thêm lock để Flask không đọc frame đang bị replace
        """
        with self.debug_frame_lock:
            if self.latest_debug_frame is None:
                return None
            return self.latest_debug_frame.copy()

    def _init_shared_camera(self) -> bool:
        try:
            self.camera = get_web_camera(self.robot.config)
            if not self.camera.is_running():
                if not self.camera.start():
                    logger.error("Camera failed to start for follow mode")
                    return False
            return True
        except Exception as e:
            logger.error(f"Camera init error for follow mode: {e}")
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
                    time.sleep(0.005)
                    continue

                _store_latest_debug_frame(self.debug_frame_lock, self, frame)

                frame_bgr = frame_to_bgr(frame, self.camera.format)

                # Detect objects (logic only, no drawing)
                detections, _ = self.detector.detect(frame_bgr)

                # Filter by target color
                target_class = self.color_map.get(self.target_color_name)
                valid_objs = [d for d in detections if d['class_name'] == target_class]

                if valid_objs:
                    # Get largest matching object
                    target = max(valid_objs, key=lambda x: x['w'] * x['h'])

                    # ===== EXTRACT TARGET INFO =====
                    frame_h, frame_w = frame_bgr.shape[:2]
                    center_x = frame_w / 2

                    self.target_x = int(target['x'])
                    self.target_y = int(target['y'])
                    self.target_w = int(target['w'])
                    self.target_h = int(target['h'])
                    self.confidence = int(target['conf'] * 100)

                    # Object size (max dimension)
                    obj_size = max(self.target_w, self.target_h)

                    # ===== DYNAMIC dt FOR BOTH FOLLOW PID CONTROLLERS =====
                    current_time = time.time()
                    if self.last_time is None:
                        dt = 0.05  # First follow frame, matches the loop sleep below
                    else:
                        dt = current_time - self.last_time
                        dt = max(0.01, min(0.2, dt))
                    self.last_time = current_time

                    # ===== PID 1: HORIZONTAL (Left/Right Centering) =====
                    # Error = target is on the LEFT → need to turn LEFT (negative error)
                    # Error = target is on the RIGHT → need to turn RIGHT (positive error)
                    error_horizontal = self.target_x - center_x
                    if abs(error_horizontal) <= self.HORIZONTAL_TOLERANCE:
                        turn_correction = 0
                        self.pid_horizontal.reset()
                    else:
                        turn_correction = self.pid_horizontal.compute(error_horizontal, dt)
                        turn_limit = min(self.TURN_SPEED_MAX, self.speed_limit)
                        turn_correction = max(-turn_limit, min(turn_limit, turn_correction))
                        turn_min = min(self.TURN_SPEED_MIN, turn_limit)
                        if turn_min > 0 and abs(turn_correction) < turn_min:
                            turn_correction = turn_min if error_horizontal > 0 else -turn_min

                    # ===== PID 2: DISTANCE (Forward/Backward) =====
                    # Error = object too small (far) → need to go FORWARD (positive error)
                    # Error = object too large (close) → need to go BACKWARD (negative error)
                    error_distance = self.TARGET_SIZE - obj_size
                    distance_correction = self.pid_distance.compute(error_distance, dt)

                    # ===== DETERMINE MOTION =====

                    # 1. Check if within dead zone (no distance adjustment needed)
                    if self.SIZE_MIN <= obj_size <= self.SIZE_MAX:
                        # Perfect size - only center horizontally
                        base_speed = 0
                        status = f"LOCKED ON {target['class_name']} ({obj_size:.0f}px) ✓"

                    elif obj_size < self.SIZE_MIN:
                        # Too small (too far) - move FORWARD using PID distance output
                        forward_max = min(self.FORWARD_SPEED_MAX, self.speed_limit)
                        forward_min = min(self.FORWARD_SPEED_MIN, forward_max)
                        requested_speed = int(max(0, min(forward_max, distance_correction)))
                        base_speed = max(forward_min, requested_speed) if forward_max > 0 else 0
                        status = f"APPROACHING {target['class_name']} ({obj_size:.0f}px) →"

                    else:
                        # Too large (too close) - move BACKWARD using PID distance output
                        backward_max = min(self.BACKWARD_SPEED, self.speed_limit)
                        backward_min = min(self.BACKWARD_SPEED_MIN, backward_max)
                        requested_speed = int(min(0, max(-backward_max, distance_correction)))
                        base_speed = min(-backward_min, requested_speed) if backward_max > 0 else 0
                        status = f"BACKING FROM {target['class_name']} ({obj_size:.0f}px) ←"

                    if base_speed < 0 and obj_size >= self.SIZE_MAX + self.BACKING_STRAIGHT_MARGIN:
                        turn_correction = 0
                        status = f"BACKING STRAIGHT FROM {target['class_name']} ({obj_size:.0f}px) ←"

                    # Keep steering from overpowering distance correction.
                    # When moving forward/backward, both wheels should keep the same direction.
                    if base_speed != 0:
                        if base_speed < 0:
                            motion_turn_limit = max(0, int(abs(base_speed) * self.BACKING_TURN_RATIO))
                        else:
                            motion_turn_limit = max(0, abs(base_speed) - 10)
                        turn_correction = max(
                            -motion_turn_limit,
                            min(motion_turn_limit, turn_correction)
                        )

                    # 2. Calculate final motor speeds
                    # Left motor: base_speed + turn_correction
                    # Right motor: base_speed - turn_correction
                    # (turn_correction < 0 → turn left, > 0 → turn right)

                    left_speed = int(base_speed + turn_correction)
                    right_speed = int(base_speed - turn_correction)

                    if base_speed < 0:
                        reverse_min = min(self.BACKWARD_SPEED_MIN, abs(base_speed), self.speed_limit)
                        if reverse_min > 0:
                            left_speed = min(left_speed, -reverse_min)
                            right_speed = min(right_speed, -reverse_min)

                    # Clamp to valid range
                    left_speed = max(-255, min(255, left_speed))
                    right_speed = max(-255, min(255, right_speed))

                    # ===== SEND TO MOTORS =====
                    # ✅ SAFETY: Dùng safe_set_motors thay vì gọi trực tiếp driver
                    if not self.robot.safe_set_motors(left_speed, right_speed):
                        logger.warning("⚠️ EMERGENCY STOP active - Follow mode blocked")
                        self.robot.current_state = 'EMERGENCY STOP'
                        time.sleep(0.1)
                        continue

                    # Update status with commanded speeds for hardware diagnostics
                    self.robot.current_state = f"{status} | L:{left_speed} R:{right_speed}"

                else:
                    # No target found - STOP and SEARCH
                    self.robot.driver.stop()
                    self.robot.current_state = f"SEARCHING {self.target_color_name.upper()}..."
                    self.confidence = 0
                    self.target_w = 0
                    self.target_h = 0
                    self.pid_horizontal.reset()
                    self.pid_distance.reset()
                    self.last_time = None

                time.sleep(0.05)

            except Exception as e:
                logger.error(f"Follow loop error: {e}")
                self.robot.driver.stop()
                break

        self.robot.driver.stop()
        logger.info("Follow loop ended")
