"""
Main Entry Point - LogisticsBot Control System
Integrates Flask Web Dashboard with Robot Hardware Control

CHANGELOG:
- FIX BUG-4: debug_feed() không còn gọi camera.capture_frame() trực tiếp trong Flask thread
  Thay bằng: đọc từ camera.capture_jpeg() → dùng buffer thread-safe của camera_manager
  Lý do: capture_frame() không thread-safe khi gọi đồng thời từ auto_loop + Flask generator
- FIX: Fallback sử dụng capture_jpeg() → convert đúng màu sắc (BGR) 
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

sys.path.append(str(Path(__file__).parent))

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
)

app = Flask(__name__)
app.config["SECRET_KEY"] = "your-secret-key-here"
socketio = SocketIO(app, cors_allowed_origins="*")

logger = setup_logger("main", "data/logs/robot.log")

robot_controller = None
auto_controller = None
follow_controller = None
motor_driver = None
config = None
LOG_FILE = "data/logs/robot.log"

follow_settings = {
    "target_color": "red",
    "follow_distance": 50,
    "tracking": False,
    "target_x": 0,
    "target_y": 0,
    "target_w": 0,
    "target_h": 0,
    "confidence": 0,
    "target_distance": 0,
}


def initialize_hardware():
    global robot_controller, auto_controller, follow_controller, motor_driver, config

    try:
        config = load_config("config/hardware_config.yaml")
        logger.info("Configuration loaded successfully")

        control_mode = config.get("control_mode", "arduino")

        if control_mode == "arduino":
            logger.info("Initializing Arduino driver...")
            arduino_config = config.get("arduino", {})

            motor_driver = ArduinoDriver(
                port=arduino_config.get("port", "/dev/ttyUSB0"),
                baudrate=arduino_config.get("baudrate", 115200),
            )

            if not motor_driver.connected:
                logger.error("Failed to connect to Arduino!")
                return False

            motor_driver.set_sensor_callback(on_arduino_sensor_data)
            logger.info("Arduino Motor Driver initialized")

        else:
            from drivers.motor.l298n_driver import L298NDriver
            logger.info("Initializing L298N driver (direct GPIO mode)...")
            motor_driver = L298NDriver(config)
            logger.info("L298N Motor Driver initialized")

        robot_controller = RobotController(motor_driver, config)
        logger.info("Robot Controller initialized")

        auto_controller = AutoModeController(robot_controller)
        logger.info("Auto Mode Controller initialized")

        follow_controller = FollowModeController(robot_controller)
        logger.info("Follow Mode Controller initialized")

        return True

    except Exception as e:
        logger.error(f"Failed to initialize hardware: {e}")
        import traceback
        traceback.print_exc()
        return False


def on_arduino_sensor_data(sensor_data: dict):
    logger.debug(f"Sensor update: {sensor_data}")
    socketio.emit("arduino_sensors", sensor_data)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """Video streaming route - dùng CameraManager's generate_frames()"""
    try:
        camera = get_web_camera(config)
        if not camera.is_running():
            if not camera.start():
                logger.error("Failed to start camera for video feed")
                return "Camera initialization failed", 500

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
    Debug video feed - ảnh màu BGR (320x240) theo chế độ hiện tại:
    - Auto Mode: Camera BGR feed
    - Follow Mode: Object detection với bounding boxes
    
    ✅ FIX BUG-4: KHÔNG gọi capture_frame() trực tiếp trong Flask thread
    → Tránh race condition với auto_loop/follow_loop đang dùng camera
    → Fallback dùng camera.capture_jpeg() (đọc từ thread-safe buffer)
    """
    def generate_debug_frames():
        while True:
            try:
                frame = None
                
                # Ưu tiên đọc từ debug buffer của controller (đã là BGR 320x240)
                if robot_controller:
                    current_mode = robot_controller.current_mode
                    
                    if current_mode == 'auto' and auto_controller:
                        # ✅ get_debug_frame() trả về COPY (thread-safe)
                        frame = auto_controller.get_debug_frame()
                    
                    elif current_mode == 'follow' and follow_controller:
                        # ✅ get_debug_frame() trả về COPY (thread-safe)
                        frame = follow_controller.get_debug_frame()
                
                # ✅ FIX BUG-4: Fallback KHÔNG gọi capture_frame() trực tiếp!
                # Thay bằng: capture_jpeg() đọc từ latest_frame buffer (thread-safe)
                # → Không tranh giành Picamera2 với auto_loop thread
                if frame is None:
                    try:
                        camera = get_web_camera(config)
                        if camera.is_running():
                            # capture_jpeg() đọc từ latest_frame buffer (đã copy, thread-safe)
                            # Trả về JPEG bytes → decode lại thành numpy array BGR
                            jpeg_bytes = camera.capture_jpeg(quality=80)
                            if jpeg_bytes is not None:
                                # Decode JPEG → numpy BGR array
                                nparr = np.frombuffer(jpeg_bytes, np.uint8)
                                frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                if frame_bgr is not None:
                                    frame = cv2.resize(frame_bgr, (320, 240))
                    except Exception as e:
                        logger.debug(f"Fallback frame error: {e}")
                
                # Placeholder nếu không có frame nào
                if frame is None:
                    frame = np.zeros((240, 320, 3), dtype=np.uint8)
                    cv2.putText(
                        frame, "Waiting for camera...", (40, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
                    )
                
                # Encode → JPEG và stream
                ret, buffer = cv2.imencode(
                    '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
                )
                if ret:
                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n'
                        + buffer.tobytes()
                        + b'\r\n'
                    )
                
                time.sleep(0.05)  # ~20 FPS
                
            except Exception as e:
                logger.error(f"Debug feed error: {e}")
                time.sleep(0.1)
    
    return Response(
        generate_debug_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/set_mode")
def set_mode():
    mode = request.args.get("mode", "auto")

    if mode not in ["auto", "follow"]:
        return jsonify({"status": "error", "message": "Invalid mode"}), 400

    if robot_controller.set_mode(mode):
        log_message(f"Mode changed to: {mode.upper()}")

        if mode == "auto":
            if auto_controller:
                auto_controller.start()
            if follow_controller:
                follow_controller.stop()
        elif mode == "follow":
            if follow_controller:
                follow_controller.start()
            if auto_controller:
                auto_controller.stop()

        socketio.emit("mode_update", {"mode": mode})
        socketio.emit("sensor_update", get_sensor_data())

        return jsonify({"status": "success", "mode": mode})
    else:
        return jsonify({"status": "error", "message": "Failed to set mode"}), 400


@app.route("/set_follow_color")
def set_follow_color():
    color = request.args.get("color", "red")

    valid_colors = ["red", "green", "blue", "yellow", "orange"]
    if color not in valid_colors:
        return jsonify({"status": "error", "message": "Invalid color"}), 400

    follow_settings["target_color"] = color
    log_message(f"Target color set to: {color.upper()}")

    if follow_controller:
        follow_controller.set_target_color(color)

    return jsonify({"status": "success", "color": color})


@app.route("/set_follow_distance")
def set_follow_distance():
    try:
        distance = int(request.args.get("distance", 50))
        distance = max(20, min(100, distance))

        follow_settings["follow_distance"] = distance
        log_message(f"Follow distance set to: {distance} cm")

        if follow_controller:
            follow_controller.set_follow_distance(distance)

        return jsonify({"status": "success", "distance": distance})
    except ValueError:
        return jsonify({"status": "error", "message": "Invalid distance value"}), 400


@app.route("/stop")
def stop():
    robot_controller.stop()
    log_message("Command: STOP")
    socketio.emit("sensor_update", get_sensor_data())
    return jsonify({"status": "success", "command": "stop"})


@app.route("/emergency_stop")
def emergency_stop():
    robot_controller.emergency_stop()
    log_message("EMERGENCY STOP executed", level="WARNING")
    socketio.emit("sensor_update", get_sensor_data())
    return jsonify({"status": "success", "command": "emergency_stop"})


@app.route("/set_speed")
def set_speed():
    try:
        speed = int(request.args.get("value", 180))
        robot_controller.set_speed(speed)
        log_message(f"Speed set to: {speed}")
        socketio.emit("sensor_update", get_sensor_data())
        return jsonify({"status": "success", "speed": speed})
    except ValueError:
        return jsonify({"status": "error", "message": "Invalid speed value"}), 400


def log_message(message: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    time_str = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {level}: {message}\n"

    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_entry)
    except Exception as e:
        logger.error(f"Error writing to log file: {e}")

    socketio.emit("log_entry", {"time": time_str, "level": level, "message": message})


@app.route("/read_log")
def read_log():
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
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write(f"[{timestamp}] INFO: Log file cleared by user\n")
        return jsonify({"status": "success", "message": "Log cleared"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


def get_sensor_data() -> dict:
    state = robot_controller.get_state()
    
    distance_value = 0.0
    line_sensors = [0] * 8
    line_pos = 0
    battery_value = 100

    if isinstance(motor_driver, ArduinoDriver):
        arduino_data = motor_driver.get_sensor_data()
        
        distance_value = arduino_data.get("distance", 0.0)
        line_sensors = arduino_data.get("line", [0] * 8)
        line_pos = arduino_data.get("line_pos", 0)
        
        if "left_speed" in arduino_data:
            state["left_motor_speed"] = arduino_data["left_speed"]
        if "right_speed" in arduino_data:
            state["right_motor_speed"] = arduino_data["right_speed"]

    return {
        "state": state["state"],
        "speed": state["speed"],
        "battery": battery_value,
        "left_motor_speed": state["left_motor_speed"],
        "right_motor_speed": state["right_motor_speed"],
        "line_sensors": line_sensors,
        "line_position": line_pos,
        "distance": distance_value,
    }


def get_target_data() -> dict:
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


@socketio.on("connect")
def handle_connect():
    logger.info("Client connected")
    emit("connection_response", {"data": "Connected"})
    emit("mode_update", {"mode": robot_controller.current_mode})
    emit("sensor_update", get_sensor_data())

    if robot_controller.current_mode == "follow":
        emit("target_update", get_target_data())


@socketio.on("disconnect")
def handle_disconnect():
    logger.info("Client disconnected")


def send_sensor_data():
    import time
    while True:
        socketio.sleep(2)
        try:
            sensor_data = get_sensor_data()
            socketio.emit("sensor_update", sensor_data)

            if robot_controller and robot_controller.current_mode == "follow":
                target_data = get_target_data()
                socketio.emit("target_update", target_data)

        except Exception as e:
            logger.error(f"Error sending sensor data: {e}")


def main():
    global LOG_FILE

    os.makedirs("data/logs", exist_ok=True)
    os.makedirs("config", exist_ok=True)

    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] INFO: LogisticsBot Control System Started\n")

    logger.info("=" * 60)
    logger.info("LogisticsBot Control System Starting...")
    logger.info("=" * 60)

    if not initialize_hardware():
        logger.error("Failed to initialize hardware. Exiting.")
        sys.exit(1)

    logger.info("Hardware initialized successfully")

    import atexit
    atexit.register(release_web_camera)
    atexit.register(lambda: robot_controller.cleanup() if robot_controller else None)

    socketio.start_background_task(send_sensor_data)

    logger.info("Starting web server on http://0.0.0.0:5000")
    logger.info("Access dashboard at: http://<raspberry-pi-ip>:5000")

    try:
        socketio.run(
            app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True
        )
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
    finally:
        release_web_camera()
        if robot_controller:
            robot_controller.cleanup()
        logger.info("Cleanup completed. Goodbye!")


if __name__ == "__main__":
    main()