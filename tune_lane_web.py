#!/usr/bin/env python3
"""
Web-based Lane Detection Tuning Tool
====================================
Công cụ tinh chỉnh thông số Lane Detection qua trình duyệt Web.
Ưu điểm: Dùng được trên điện thoại, không cần màn hình HDMI.

Cách dùng:
1. Chạy: python3 tools/tune_lane_web.py
2. Mở trình duyệt: http://<IP_CUA_PI>:5000
"""

import cv2
import numpy as np
import yaml
import time
import threading
import sys
import os
import logging
from pathlib import Path
from flask import Flask, render_template_string, Response, request, jsonify

# Thêm đường dẫn để import module dự án
sys.path.append(str(Path(__file__).parent.parent))
from perception.camera_manager import get_web_camera, yuv420_to_bgr

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Khởi tạo Flask App
app = Flask(__name__)

# Biến toàn cục
config_path = 'config/hardware_config.yaml'
camera = None
lock = threading.Lock()

DEFAULT_PARAMS = {
    'roi_top_ratio': 0.6,
    'roi_bottom_ratio': 1.0,
    'roi_width_top': 30,  # Phần trăm thu hẹp cạnh trên (0-50%)
    'canny_low': 80,
    'canny_high': 185,
    'hough_threshold': 40,
    'blur_kernel': 7,
}

PARAM_RANGES = {
    'roi_top_ratio': {'min': 0.0, 'max': 0.99, 'type': float},
    'roi_bottom_ratio': {'min': 0.01, 'max': 1.0, 'type': float},
    'roi_width_top': {'min': 0, 'max': 50, 'type': int},
    'canny_low': {'min': 0, 'max': 1000, 'type': int},
    'canny_high': {'min': 0, 'max': 1000, 'type': int},
    'hough_threshold': {'min': 1, 'max': 1000, 'type': int},
    'blur_kernel': {'min': 1, 'max': 99, 'type': int},
}

CONFIG_SAVE_KEYS = (
    'roi_top_ratio',
    'roi_bottom_ratio',
    'canny_low',
    'canny_high',
    'hough_threshold',
    'blur_kernel',
)


def load_config():
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        if not isinstance(config, dict):
            logger.error("Config file %s is not a YAML mapping", config_path)
            return {}
        return config
    return {}


def load_lane_params(full_cfg):
    """Load tuning params from ai.lane_detection, falling back to legacy config."""
    params = DEFAULT_PARAMS.copy()
    ai_lane = full_cfg.get('ai', {}).get('lane_detection', {})
    legacy_lane = full_cfg.get('lane_following', {}).get('lane_detection', {})

    if isinstance(ai_lane, dict) and ai_lane:
        params.update(ai_lane)
    elif isinstance(legacy_lane, dict) and legacy_lane:
        logger.warning(
            "Using legacy lane_following.lane_detection; save will write ai.lane_detection"
        )
        params.update(legacy_lane)

    normalized = {}
    for key, default_value in DEFAULT_PARAMS.items():
        normalized[key] = validate_param(key, params.get(key, default_value))
    return normalized


def validate_param(key, value):
    """Validate and normalize one tuning parameter."""
    if key not in PARAM_RANGES:
        raise ValueError(f"Invalid parameter: {key}")

    if isinstance(value, bool):
        raise ValueError(f"{key} must be numeric")

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be numeric")

    if not np.isfinite(numeric_value):
        raise ValueError(f"{key} must be finite")

    param_range = PARAM_RANGES[key]
    numeric_value = max(param_range['min'], min(param_range['max'], numeric_value))

    if param_range['type'] is int:
        int_value = int(round(numeric_value))
        if key == 'blur_kernel':
            if int_value % 2 == 0:
                int_value += 1
            if int_value > param_range['max']:
                int_value -= 2
            int_value = max(param_range['min'], int_value)
        return int_value

    return float(numeric_value)


def _looks_like_yuv420(frame):
    if frame is None or len(frame.shape) != 2:
        return False
    height, width = frame.shape[:2]
    if height % 3 != 0 or width % 2 != 0:
        return False
    image_height = (height * 2) // 3
    if image_height <= 0 or image_height % 2 != 0:
        return False
    aspect_ratio = image_height / float(width)
    return 0.55 <= aspect_ratio <= 1.8


def frame_to_bgr(frame, camera_format=None):
    """Convert captured CameraManager frame to BGR for OpenCV debug processing."""
    if frame is None or not hasattr(frame, 'shape'):
        raise ValueError("Invalid frame")

    format_name = str(camera_format or '').upper()
    if len(frame.shape) == 2:
        if format_name == 'YUV420' or _looks_like_yuv420(frame):
            return yuv420_to_bgr(frame)
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    if len(frame.shape) == 3 and frame.shape[2] == 3:
        if format_name.startswith('RGB'):
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    raise ValueError(f"Unsupported frame shape: {frame.shape}")


full_config = load_config()
current_params = load_lane_params(full_config)

# HTML Template (Giao diện Web)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🚗 Robot Car Lane Tuning</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: sans-serif; background: #1a1a1a; color: #fff; margin: 0; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; display: flex; flex-wrap: wrap; gap: 20px; }
        .video-box { flex: 2; min-width: 320px; background: #000; border-radius: 10px; overflow: hidden; }
        .controls { flex: 1; min-width: 300px; background: #2d2d2d; padding: 20px; border-radius: 10px; }
        img { width: 100%; display: block; }
        .slider-group { margin-bottom: 15px; }
        label { display: flex; justify-content: space-between; margin-bottom: 5px; font-weight: bold; }
        input[type=range] { width: 100%; height: 10px; border-radius: 5px; background: #555; outline: none; }
        .btn { width: 100%; padding: 15px; border: none; border-radius: 5px; background: #4CAF50; color: white; font-size: 16px; cursor: pointer; margin-top: 20px; }
        .btn:hover { background: #45a049; }
        .value-display { color: #4CAF50; }
        h2 { margin-top: 0; border-bottom: 1px solid #555; padding-bottom: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="video-box">
            <img src="{{ url_for('video_feed') }}" alt="Video Stream">
        </div>
        <div class="controls">
            <h2>🎛️ Tuning Parameters</h2>
            
            <div class="slider-group">
                <label>Canny Low: <span id="val_canny_low">{{ params.canny_low }}</span></label>
                <input type="range" min="0" max="255" value="{{ params.canny_low }}" oninput="update('canny_low', this.value)">
            </div>

            <div class="slider-group">
                <label>Canny High: <span id="val_canny_high">{{ params.canny_high }}</span></label>
                <input type="range" min="0" max="255" value="{{ params.canny_high }}" oninput="update('canny_high', this.value)">
            </div>

            <hr style="border-color: #555;">

            <div class="slider-group">
                <label>ROI Top (%): <span id="val_roi_top_ratio">{{ (params.roi_top_ratio * 100)|int }}</span></label>
                <input type="range" min="10" max="90" value="{{ (params.roi_top_ratio * 100)|int }}" oninput="update('roi_top_ratio', this.value/100)">
            </div>
            
            <div class="slider-group">
                <label>ROI Trap Width (%): <span id="val_roi_width_top">{{ params.roi_width_top }}</span></label>
                <input type="range" min="0" max="50" value="{{ params.roi_width_top }}" oninput="update('roi_width_top', this.value)">
            </div>

            <hr style="border-color: #555;">

            <div class="slider-group">
                <label>Hough Threshold: <span id="val_hough_threshold">{{ params.hough_threshold }}</span></label>
                <input type="range" min="10" max="200" value="{{ params.hough_threshold }}" oninput="update('hough_threshold', this.value)">
            </div>

            <button class="btn" onclick="saveConfig()">💾 SAVE TO CONFIG</button>
        </div>
    </div>

    <script>
        function update(key, value) {
            // Cập nhật số hiển thị
            let displayVal = value;
            if (key === 'roi_top_ratio') displayVal = Math.round(value * 100);
            document.getElementById('val_' + key).innerText = displayVal;

            // Gửi dữ liệu về server
            fetch('/update_params', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({[key]: parseFloat(value)})
            });
        }

        function saveConfig() {
            fetch('/save_config', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    if(data.success) alert("✅ Saved to hardware_config.yaml!");
                    else alert("❌ Error saving config!");
                });
        }
    </script>
</body>
</html>
"""

def process_frame(frame):
    """Xử lý ảnh để hiển thị debug"""
    with lock:
        params = current_params.copy()
    
    # 1. Resize cho nhẹ
    frame = cv2.resize(frame, (640, 480))
    height, width = frame.shape[:2]
    
    # 2. Xử lý ảnh (Canny)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (params['blur_kernel'], params['blur_kernel']), 0)
    edges = cv2.Canny(blur, params['canny_low'], params['canny_high'])
    
    # 3. Tạo ROI (Hình thang)
    roi_top = int(height * params['roi_top_ratio'])
    roi_bot = int(height * params['roi_bottom_ratio'])
    w_margin = int(width * (params['roi_width_top'] / 100.0))
    
    vertices = np.array([[
        (0, roi_bot),
        (w_margin, roi_top),
        (width - w_margin, roi_top),
        (width, roi_bot)
    ]], dtype=np.int32)
    
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # 4. Tìm đường thẳng (Hough) - Mô phỏng
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 
                           int(params['hough_threshold']), 
                           minLineLength=50, maxLineGap=30)
    
    # 5. Vẽ Debug
    debug_img = frame.copy()
    
    # Vẽ khung ROI (Màu Vàng)
    cv2.polylines(debug_img, [vertices], True, (0, 255, 255), 2)
    
    # Vẽ Lines tìm được (Màu Đỏ)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
    # Ghép ảnh: Trái (Edges) - Phải (Debug màu)
    edges_bgr = cv2.cvtColor(masked_edges, cv2.COLOR_GRAY2BGR)
    combined = np.hstack((edges_bgr, debug_img))
    
    return combined

def generate_frames():
    global camera
    while True:
        if camera is None:
            break

        try:
            if not camera.is_running():
                if not camera.start():
                    logger.error("Camera failed to start for tuning video feed")
                    time.sleep(0.5)
                    continue

            frame = camera.capture_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            frame_bgr = frame_to_bgr(frame, getattr(camera, 'format', None))
            processed_frame = process_frame(frame_bgr)

            # Encode JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if not ret:
                logger.error("Could not encode tuning frame as JPEG")
                time.sleep(0.05)
                continue
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except GeneratorExit:
            logger.info("Tuning video client disconnected")
            break
        except Exception as e:
            logger.error(f"Tuning video frame error: {e}")
            time.sleep(0.1)

# ===== FLASK ROUTES =====
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, params=current_params)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update_params', methods=['POST'])
def update_params():
    global current_params
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(success=False, message="Invalid JSON payload"), 400

    updates = {}
    try:
        for key, value in data.items():
            updates[key] = validate_param(key, value)
    except ValueError as e:
        return jsonify(success=False, message=str(e)), 400

    with lock:
        current_params.update(updates)

    return jsonify(success=True, params=updates)

@app.route('/save_config', methods=['POST'])
def save_config():
    try:
        # Load lại file gốc để tránh mất các config khác
        full_cfg = load_config()
        if not isinstance(full_cfg, dict):
            full_cfg = {}

        with lock:
            params = current_params.copy()
            
        # Cập nhật section chính ai.lane_detection
        lane_section = full_cfg.setdefault('ai', {}).setdefault('lane_detection', {})
        for key in CONFIG_SAVE_KEYS:
            lane_section[key] = params[key]

        # Backward compatibility: mirror legacy section only if it already exists.
        legacy_section = full_cfg.get('lane_following', {}).get('lane_detection')
        if isinstance(legacy_section, dict):
            for key in CONFIG_SAVE_KEYS:
                legacy_section[key] = params[key]
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(full_cfg, f, default_flow_style=False, allow_unicode=True)
            
        logger.info("Config saved successfully to %s", config_path)
        return jsonify(success=True)
    except Exception as e:
        logger.error(f"Error saving config: {e}")
        return jsonify(success=False, message=str(e)), 500

# ===== MAIN =====
if __name__ == '__main__':
    # Init Camera từ Project
    logger.info("Initializing CameraManager...")
    config = load_config()
    camera = get_web_camera(config)
    
    if not camera.is_running():
        if not camera.start():
            logger.error("Camera failed to start for tuning tool")
        
    logger.info("Web Tuning Tool started at http://0.0.0.0:5000")
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        if camera:
            camera.stop()
