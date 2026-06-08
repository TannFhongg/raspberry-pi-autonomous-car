"""
Main Entry Point - LogisticsBot Control System
Integrates Flask Web Dashboard with Robot Hardware Control
Supports Arduino Nano for motor/sensor control
UPDATED: Picamera2 support for video streaming
"""

from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
from datetime import datetime
import cv2
import numpy as np
import yaml
import logging
import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import custom modules
from drivers.motor.arduino_driver import ArduinoDriver
from control.robot_controller import (
    RobotController,
    AutoModeController,
    FollowModeController,
)
from utils.logger import setup_logger
from utils.config_loader import load_config
from perception.camera_manager import (
    get_web_camera,
    release_web_camera,
    yuv420_to_bgr,
)  # NEW: Picamera2

# Initialize Flask app
app = Flask(__name__)
app.config["SECRET_KEY"] = "your-secret-key-here"
socketio = SocketIO(app, cors_allowed_origins="*")

# Setup logging
logger = setup_logger("main", "data/logs/robot.log")

# Global variables
robot_controller = None
auto_controller = None
follow_controller = None
motor_driver = None
config = None
LOG_FILE = "data/logs/robot.log"

# Follow mode settings
follow_settings = {
    "target_color": "red",
    "tracking": False,
    "target_x": 0,
    "target_y": 0,
    "target_w": 0,
    "target_h": 0,
    "confidence": 0,
    "target_distance": 0,
}


def initialize_hardware():
    """Initialize robot hardware"""
    global robot_controller, auto_controller, follow_controller, motor_driver, config

    try:
        # Load configuration
        config = load_config("config/hardware_config.yaml")
        logger.info("Configuration loaded successfully")

        # Determine control mode
        control_mode = config.get("control_mode", "arduino")

        if control_mode == "arduino":
            # Use Arduino for motor control
            logger.info("Initializing Arduino driver...")
            arduino_config = config.get("arduino", {})

            motor_driver = ArduinoDriver(
                port=arduino_config.get("port", "/dev/ttyUSB0"),
                baudrate=arduino_config.get("baudrate", 115200),
            )

            if not motor_driver.connected:
                logger.error("Failed to connect to Arduino!")
                return False

            # Set sensor callback
            motor_driver.set_sensor_callback(on_arduino_sensor_data)
            logger.info("Arduino Motor Driver initialized")

        else:
            logger.error(
                "Unsupported control_mode '%s'. Direct GPIO mode requires "
                "drivers/motor/l298n_driver.py, which is not present. "
                "Set control_mode: 'arduino' in config/hardware_config.yaml.",
                control_mode,
            )
            return False

        # Initialize robot controller
        robot_controller = RobotController(motor_driver, config)
        logger.info("Robot Controller initialized")

        # Initialize auto mode controller
        auto_controller = AutoModeController(robot_controller)
        logger.info("Auto Mode Controller initialized")

        # Initialize follow controller
        follow_controller = FollowModeController(robot_controller)
        logger.info("Follow Mode Controller initialized")

        return True

    except Exception as e:
        logger.error(f"Failed to initialize hardware: {e}")
        import traceback

        traceback.print_exc()
        return False


def on_arduino_sensor_data(sensor_data: dict):
    """
    Callback when Arduino sends sensor data

    Args:
        sensor_data: Dictionary with sensor readings
    """
    # Update global sensor data for web interface
    # This is called automatically by Arduino driver
    logger.debug(f"Sensor update: {sensor_data}")

    # Emit to all connected clients
    socketio.emit("arduino_sensors", sensor_data)


# ===== FLASK ROUTES =====


@app.route("/")
def index():
    """Render main dashboard"""
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """
    Video streaming route with Picamera2
    UPDATED: Now uses CameraManager with Picamera2
    """
    try:
        # Get global camera instance
        camera = get_web_camera(config)

        # Start camera if not running
        if not camera.is_running():
            if not camera.start():
                logger.error("Failed to start camera for video feed")
                return "Camera initialization failed", 500

        # Return streaming response
        return Response(
            camera.generate_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    except Exception as e:
        logger.error(f"Video feed error: {e}")
        import traceback

        traceback.print_exc()
        return "Camera error", 500


@app.route("/debug_feed")
def debug_feed():
    """
    Debug video feed - shows BGR color frames (320x240) based on current mode
    - Auto Mode: Clean camera BGR feed from lane detection
    - Follow Mode: Clean camera BGR feed without detection overlays
    """
    def generate_debug_frames():
        # Ensure camera is started
        try:
            camera = get_web_camera(config)
            if not camera.is_running():
                camera.start()
        except Exception as e:
            logger.error(f"Failed to init camera for debug feed: {e}")

        while True:
            try:
                frame = None
                current_mode = None

                # Get debug frame based on current mode
                if robot_controller:
                    current_mode = robot_controller.current_mode

                    if current_mode == 'auto' and auto_controller:
                        # Auto mode: BGR frame from camera (resized 320x240)
                        frame = auto_controller.get_debug_frame()

                    elif current_mode == 'follow' and follow_controller:
                        # Follow mode: Clean BGR frame from camera (resized 320x240)
                        frame = follow_controller.get_debug_frame()

                # In idle/startup, capture directly for a live clean feed.
                # In active modes, avoid direct capture because control loops own it.
                if frame is None:
                    try:
                        camera = get_web_camera(config)
                        if camera.is_running():
                            if frame is None and current_mode not in ['auto', 'follow']:
                                frame_yuv = camera.capture_frame()
                                if frame_yuv is not None:
                                    frame_bgr = yuv420_to_bgr(frame_yuv)
                                    frame = cv2.resize(frame_bgr, (320, 240))

                            if frame is None:
                                jpeg_bytes = camera.capture_jpeg(quality=80)
                                if jpeg_bytes is not None:
                                    nparr = np.frombuffer(jpeg_bytes, np.uint8)
                                    frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                    if frame_bgr is not None:
                                        frame = cv2.resize(frame_bgr, (320, 240))
                    except Exception:
                        pass

                # Create placeholder if still no frame
                if frame is None:
                    frame = np.zeros((240, 320, 3), dtype=np.uint8)
                    cv2.putText(frame, "Waiting for camera...", (50, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                # Encode to JPEG
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

                time.sleep(0.02)  # ~20 FPS

            except Exception as e:
                logger.error(f"Debug feed error: {e}")
                time.sleep(0.1)

    return Response(
        generate_debug_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ===== MODE CONTROL =====


@app.route("/set_mode")
def set_mode():
    """Set control mode (auto/follow/idle - manual mode removed)"""
    mode = request.args.get("mode", "idle")

    if mode not in ["auto", "follow", "idle"]:
        return jsonify({"status": "error", "message": "Invalid mode"}), 400

    if robot_controller is None:
        logger.error("Mode switch rejected: robot controller is not initialized")
        return jsonify({"status": "error", "message": "Robot controller unavailable"}), 503

    previous_mode = robot_controller.current_mode
    previous_state = robot_controller.current_state

    if mode == "idle":
        if auto_controller:
            auto_controller.stop()
        if follow_controller:
            follow_controller.stop()
        robot_controller.set_mode("idle")
        robot_controller.stop()
        log_message("Mode changed to: IDLE")
        socketio.emit("mode_update", {"mode": "idle"})
        socketio.emit("sensor_update", get_sensor_data())
        return jsonify({"status": "success", "mode": "idle"})

    target_controller = auto_controller if mode == "auto" else follow_controller
    other_controller = follow_controller if mode == "auto" else auto_controller

    if target_controller is None:
        logger.error(f"Mode switch failed: {mode} controller is not initialized")
        return jsonify({"status": "error", "message": f"{mode} controller unavailable"}), 503

    if not robot_controller.set_mode(mode):
        return jsonify({"status": "error", "message": "Failed to set mode"}), 400

    started = bool(getattr(target_controller, "running", False))
    if not started:
        try:
            started = target_controller.start()
        except Exception as e:
            logger.error(f"Mode switch to {mode} raised during start: {e}")
            started = False

    if not started:
        logger.error(
            "Mode switch to %s failed. Camera/controller did not start. "
            "Rolling back from previous mode=%s state=%s.",
            mode,
            previous_mode,
            previous_state,
        )
        try:
            target_controller.stop()
        except Exception as e:
            logger.error(f"Error stopping failed {mode} controller: {e}")

        if auto_controller:
            auto_controller.stop()
        if follow_controller:
            follow_controller.stop()
        robot_controller.set_mode("idle")
        robot_controller.stop()

        message = f"Failed to start {mode} mode. Check camera/controller logs."
        log_message(message, level="ERROR")
        socketio.emit("mode_update", {"mode": "idle"})
        socketio.emit("sensor_update", get_sensor_data())
        return jsonify({"status": "error", "message": message, "mode": "idle"}), 503

    if other_controller:
        other_controller.stop()

    log_message(f"Mode changed to: {mode.upper()}")
    socketio.emit("mode_update", {"mode": mode})
    socketio.emit("sensor_update", get_sensor_data())
    return jsonify({"status": "success", "mode": mode})


# ===== FOLLOW MODE SETTINGS =====


@app.route("/set_follow_color")
def set_follow_color():
    """Set target color for follow mode"""
    color = request.args.get("color", "red")

    valid_colors = ["red", "green", "blue", "yellow"]
    if color not in valid_colors:
        return jsonify({"status": "error", "message": "Invalid color"}), 400

    if follow_controller is None:
        logger.error("Follow color rejected: follow controller is not initialized")
        return jsonify({"status": "error", "message": "Follow controller unavailable"}), 503

    follow_settings["target_color"] = color
    log_message(f"Target color set to: {color.upper()}")

    follow_controller.set_target_color(color)

    return jsonify({"status": "success", "color": color})


# ===== ROBOT CONTROL COMMANDS (Manual mode removed) =====


@app.route("/stop")
def stop():
    """Stop controllers and return to standby"""
    if robot_controller is None:
        logger.error("Stop rejected: robot controller is not initialized")
        return jsonify({"status": "error", "message": "Robot controller unavailable"}), 503

    robot_controller.set_mode("idle")
    if auto_controller:
        auto_controller.stop()
    if follow_controller:
        follow_controller.stop()
    robot_controller.stop()
    log_message("Command: STANDBY")
    socketio.emit("mode_update", {"mode": "idle"})
    socketio.emit("sensor_update", get_sensor_data())
    return jsonify({"status": "success", "command": "stop", "mode": "idle"})


@app.route("/emergency_stop")
def emergency_stop():
    """Emergency stop"""
    if robot_controller is None:
        logger.error("Emergency stop rejected: robot controller is not initialized")
        return jsonify({"status": "error", "message": "Robot controller unavailable"}), 503

    robot_controller.emergency_stop()
    log_message("EMERGENCY STOP executed", level="WARNING")
    socketio.emit("sensor_update", get_sensor_data())
    return jsonify({"status": "success", "command": "emergency_stop"})


# ===== SPEED CONTROL =====


@app.route("/set_speed")
def set_speed():
    """Set motor speed"""
    if robot_controller is None:
        logger.error("Speed update rejected: robot controller is not initialized")
        return jsonify({"status": "error", "message": "Robot controller unavailable"}), 503

    try:
        speed = int(request.args.get("value", 180))
        if speed < 0 or speed > 255:
            return jsonify({"status": "error", "message": "Speed must be 0-255"}), 400

        applied_speed = robot_controller.set_speed(speed)
        if auto_controller:
            auto_controller.set_speed(applied_speed)
        if follow_controller:
            follow_controller.set_speed(applied_speed)

        log_message(f"Speed set to: {applied_speed}")
        socketio.emit("sensor_update", get_sensor_data())
        return jsonify({"status": "success", "speed": applied_speed})
    except ValueError:
        return jsonify({"status": "error", "message": "Invalid speed value"}), 400


# ===== LOG MANAGEMENT =====


def log_message(message: str, level: str = "INFO"):
    """Log message and emit to clients"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    time_str = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {level}: {message}\n"

    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_entry)
    except Exception as e:
        logger.error(f"Error writing to log file: {e}")

    # Emit to clients
    socketio.emit("log_entry", {"time": time_str, "level": level, "message": message})


@app.route("/read_log")
def read_log():
    """Read log file"""
    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                content = f.read()
            return content, 200
        else:
            return "Log file not found", 404
    except Exception as e:
        return f"Error reading log: {str(e)}", 500


@app.route("/clear_log")
def clear_log():
    """Clear log file"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write(f"[{timestamp}] INFO: Log file cleared by user\n")
        return jsonify({"status": "success", "message": "Log cleared"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500






# ===== SENSOR DATA =====


def get_sensor_data() -> dict:
    """Get current sensor/robot state"""
    if robot_controller is None:
        return {
            "state": "CONTROLLER UNAVAILABLE",
            "speed": 0,
            "battery": 0,
            "left_motor_speed": 0,
            "right_motor_speed": 0,
            "line_sensors": [0] * 8,
            "line_position": 0,
            "distance": 0.0,
        }

    # Lấy trạng thái từ robot controller (đã bao gồm tốc độ motor)
    state = robot_controller.get_state()

    # Mặc định (nếu không có dữ liệu)
    distance_value = 0.0
    line_sensors = [0] * 8
    line_pos = 0

    # Pin (Hiện tại chưa có cảm biến pin, nên để cố định hoặc 0 thay vì random)
    # Bạn có thể sửa thành 100 hoặc 0 tùy ý để biết đây là giá trị giả định
    battery_value = 100

    # Lấy dữ liệu THẬT từ Arduino (nếu đang dùng chế độ Arduino)
    if isinstance(motor_driver, ArduinoDriver):
        arduino_data = motor_driver.get_sensor_data()

        # Lấy giá trị thực từ phần cứng
        distance_value = arduino_data.get("distance", 0.0)
        line_sensors = arduino_data.get("line", [0] * 8)
        line_pos = arduino_data.get("line_pos", 0)

        # Cập nhật tốc độ thực tế từ Arduino (nếu có)
        if "left_speed" in arduino_data:
            state["left_motor_speed"] = arduino_data["left_speed"]
        if "right_speed" in arduino_data:
            state["right_motor_speed"] = arduino_data["right_speed"]

    return {
        "state": state["state"],
        "speed": state["speed"],
        "battery": battery_value,  # Không còn random
        "left_motor_speed": state["left_motor_speed"],
        "right_motor_speed": state["right_motor_speed"],
        "line_sensors": line_sensors,
        "line_position": line_pos,
        "distance": distance_value, # Dữ liệu thật 100% hoặc 0.0
    }


def get_target_data() -> dict:
    """Get current target tracking data for follow mode"""
    if follow_controller:
        return follow_controller.get_target_data()

    return {
        "tracking": follow_settings["tracking"],
        "target_color": follow_settings["target_color"],
        "target_x": follow_settings["target_x"],
        "target_y": follow_settings["target_y"],
        "target_w": follow_settings["target_w"],
        "target_h": follow_settings["target_h"],
        "confidence": follow_settings["confidence"],
        "target_distance": follow_settings["target_distance"],
    }


# ===== SOCKETIO EVENTS =====


@socketio.on("connect")
def handle_connect():
    """Handle client connection"""
    logger.info("Client connected")
    emit("connection_response", {"data": "Connected"})

    # Send current mode and state
    mode = robot_controller.current_mode if robot_controller else "idle"
    emit("mode_update", {"mode": mode})
    emit("sensor_update", get_sensor_data())

    # Send target data if in follow mode
    if robot_controller and robot_controller.current_mode == "follow":
        emit("target_update", get_target_data())


@socketio.on("disconnect")
def handle_disconnect():
    """Handle client disconnection"""
    logger.info("Client disconnected")


# ===== BACKGROUND TASKS =====


def send_sensor_data():
    """Send sensor data periodically to all clients"""
    import time

    while True:
        socketio.sleep(2)

        try:
            sensor_data = get_sensor_data()
            socketio.emit("sensor_update", sensor_data)

            # Send target data if in follow mode
            if robot_controller and robot_controller.current_mode == "follow":
                target_data = get_target_data()
                socketio.emit("target_update", target_data)

        except Exception as e:
            logger.error(f"Error sending sensor data: {e}")


# ===== MAIN =====


def main():
    """Main entry point"""
    global LOG_FILE

    # Create directories if not exist
    os.makedirs("data/logs", exist_ok=True)
    os.makedirs("config", exist_ok=True)

    # Initialize log file
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] INFO: LogisticsBot Control System Started\n")

    logger.info("=" * 60)
    logger.info("LogisticsBot Control System Starting...")
    logger.info("=" * 60)

    # Initialize hardware
    if not initialize_hardware():
        logger.error("Failed to initialize hardware. Exiting.")
        logger.error("Please check:")
        logger.error("  1. Arduino is connected to USB port")
        logger.error("  2. Serial port is correct in hardware_config.yaml")
        logger.error("  3. User has permission to access serial port")
        logger.error("     Run: sudo usermod -a -G dialout $USER")
        sys.exit(1)

    logger.info("Hardware initialized successfully")

    # Register cleanup on exit
    import atexit

    atexit.register(release_web_camera)
    atexit.register(lambda: robot_controller.cleanup() if robot_controller else None)

    # Start background task
    socketio.start_background_task(send_sensor_data)

    # Run Flask-SocketIO server
    logger.info("Starting web server on http://0.0.0.0:5000")
    logger.info("Access dashboard at: http://<raspberry-pi-ip>:5000")

    try:
        socketio.run(
            app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True
        )
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
    finally:
        # Cleanup
        release_web_camera()
        if robot_controller:
            robot_controller.cleanup()
        logger.info("Cleanup completed. Goodbye!")


if __name__ == "__main__":
    main()
