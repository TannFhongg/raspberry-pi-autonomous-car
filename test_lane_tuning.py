#!/usr/bin/env python3
"""
Lane Detection Tuning Tool - Flask Web Version
===============================================
Tool để tune các thông số lane detection qua Web Browser (SSH Remote)

Cách sử dụng:
1. Chạy: python test_lane_tuning.py
2. Mở browser: http://<raspberry_pi_ip>:5000
3. Điều chỉnh thông số qua giao diện web
4. Xem kết quả real-time

Features:
- Live video stream với lane detection
- Điều chỉnh thông số real-time qua sliders
- Lưu/Load config
- Chụp ảnh để phân tích
"""

from flask import Flask, render_template_string, Response, jsonify, request
from picamera2 import Picamera2
import cv2
import numpy as np
import logging
import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime

# Thêm path để import các module
sys.path.append(str(Path(__file__).parent))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== CẤU HÌNH MẶC ĐỊNH =====
DEFAULT_CONFIG = {
    'roi_top_ratio': 0.35,
    'roi_bottom_ratio': 1.0,
    'canny_low': 40,
    'canny_high': 120,
    'hough_threshold': 20,
    'min_line_length': 30,
    'max_line_gap': 20,
    'blur_kernel': 5,
}

CONFIG_FILE = "lane_config.json"
OUTPUT_DIR = "output_lane_tuning"

# ===== FLASK APP =====
app = Flask(__name__)

# Global variables
camera = None
current_config = DEFAULT_CONFIG.copy()
latest_error = 0
latest_x_line = 0
frame_count = 0


def init_camera():
    """Khởi tạo Picamera2"""
    global camera
    try:
        camera = Picamera2()
        config = camera.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        camera.configure(config)
        camera.start()
        time.sleep(1)  # Chờ camera ổn định
        logger.info("✅ Camera initialized: 640x480")
        return True
    except Exception as e:
        logger.error(f"❌ Camera init error: {e}")
        return False


def load_config():
    """Load config từ file"""
    global current_config
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                saved = json.load(f)
                current_config.update(saved)
            logger.info(f"✅ Config loaded from {CONFIG_FILE}")
        except Exception as e:
            logger.error(f"⚠️ Could not load config: {e}")


def save_config():
    """Lưu config ra file"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(current_config, f, indent=2)
        logger.info(f"✅ Config saved to {CONFIG_FILE}")
        return True
    except Exception as e:
        logger.error(f"❌ Could not save config: {e}")
        return False


def detect_lane(frame, config):
    """
    Phát hiện lane với config tùy chỉnh
    Trả về: error, x_line, center_x, debug_frame
    """
    global latest_error, latest_x_line
    
    height, width = frame.shape[:2]
    center_x = width // 2
    LANE_WIDTH_PIXELS = 310
    
    # Debug frame
    frame_debug = frame.copy()
    cv2.line(frame_debug, (center_x, 0), (center_x, height), (0, 255, 255), 2)
    
    # 1. Grayscale + Invert
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray_inv = cv2.bitwise_not(gray)
    
    # 2. Blur
    kernel_size = config['blur_kernel']
    if kernel_size % 2 == 0:
        kernel_size += 1
    blur = cv2.GaussianBlur(gray_inv, (kernel_size, kernel_size), 0)
    
    # 3. CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(blur)
    
    # 4. Canny
    edges = cv2.Canny(enhanced, config['canny_low'], config['canny_high'])
    
    # 5. ROI
    roi_top = int(height * config['roi_top_ratio'])
    roi_bottom = int(height * config['roi_bottom_ratio'])
    
    roi_vertices = np.array([[
        (0, roi_bottom),
        (int(width * 0.30), roi_top),
        (int(width * 0.70), roi_top),
        (width, roi_bottom)
    ]], dtype=np.int32)
    
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Vẽ ROI
    cv2.polylines(frame_debug, roi_vertices, True, (255, 0, 0), 2)
    
    # 6. Hough Transform
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=config['hough_threshold'],
        minLineLength=config['min_line_length'],
        maxLineGap=config['max_line_gap']
    )
    
    # 7. Phân loại lines
    left_lines = []
    right_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) < 1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.5:
                continue
            mid_x = (x1 + x2) / 2
            if slope < -0.5 and mid_x < center_x:
                left_lines.append((x1, y1, x2, y2, slope))
            elif slope > 0.5 and mid_x > center_x:
                right_lines.append((x1, y1, x2, y2, slope))
    
    # 8. Tính vị trí lane
    def calc_lane_x(lines, color):
        if not lines:
            return None
        x_bottoms = []
        for x1, y1, x2, y2, slope in lines:
            x_bottom = x1 + (height - y1) / slope
            if 0 <= x_bottom <= width:
                x_bottoms.append(x_bottom)
                cv2.line(frame_debug, (x1, y1), (x2, y2), color, 2)
        return int(np.median(x_bottoms)) if x_bottoms else None
    
    left_x = calc_lane_x(left_lines, (0, 255, 0))
    right_x = calc_lane_x(right_lines, (255, 0, 0))
    
    # 9. Tính tâm đường
    if left_x is not None and right_x is not None:
        x_line = (left_x + right_x) // 2
        status = "BOTH_LANES"
        cv2.circle(frame_debug, (left_x, height - 10), 10, (0, 255, 0), -1)
        cv2.circle(frame_debug, (right_x, height - 10), 10, (255, 0, 0), -1)
    elif left_x is not None:
        x_line = left_x + (LANE_WIDTH_PIXELS // 2)
        status = "LEFT_ONLY"
        cv2.circle(frame_debug, (left_x, height - 10), 10, (0, 255, 0), -1)
    elif right_x is not None:
        x_line = right_x - (LANE_WIDTH_PIXELS // 2)
        status = "RIGHT_ONLY"
        cv2.circle(frame_debug, (right_x, height - 10), 10, (255, 0, 0), -1)
    else:
        x_line = center_x
        status = "NO_LANE"
    
    error = x_line - center_x
    latest_error = error
    latest_x_line = x_line
    
    # Vẽ kết quả
    cv2.line(frame_debug, (x_line, 0), (x_line, height), (255, 0, 255), 3)
    cv2.arrowedLine(frame_debug, (center_x, height - 50), (x_line, height - 50), (0, 0, 255), 4)
    
    # Text info
    cv2.putText(frame_debug, f"Error: {error:+d}px | {status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame_debug, f"L:{left_x} R:{right_x}", (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    return error, x_line, center_x, frame_debug, edges


def generate_frames():
    """Generator cho video stream"""
    global frame_count
    while True:
        if camera is None:
            time.sleep(0.1)
            continue
        
        try:
            frame = camera.capture_array()
            frame_count += 1
            
            # Detect lane
            error, x_line, center_x, debug_frame, edges = detect_lane(frame, current_config)
            
            # Convert RGB to BGR for OpenCV encoding
            debug_bgr = cv2.cvtColor(debug_frame, cv2.COLOR_RGB2BGR)
            
            # Encode to JPEG
            ret, buffer = cv2.imencode('.jpg', debug_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS
            
        except Exception as e:
            logger.error(f"Frame error: {e}")
            time.sleep(0.1)


def generate_edges():
    """Generator cho edges stream"""
    while True:
        if camera is None:
            time.sleep(0.1)
            continue
        
        try:
            frame = camera.capture_array()
            _, _, _, _, edges = detect_lane(frame, current_config)
            
            # Convert to 3 channel for display
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
            ret, buffer = cv2.imencode('.jpg', edges_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.05)
            
        except Exception as e:
            time.sleep(0.1)


# ===== HTML TEMPLATE =====
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>🚗 Lane Tuning Tool</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: 'Segoe UI', Arial, sans-serif; 
            background: #1a1a2e; 
            color: #eee; 
            padding: 10px;
        }
        h1 { 
            text-align: center; 
            color: #00d4ff; 
            margin-bottom: 15px;
            font-size: 1.5em;
        }
        .container { 
            display: flex; 
            flex-wrap: wrap; 
            gap: 15px; 
            max-width: 1400px; 
            margin: 0 auto;
        }
        .video-section { 
            flex: 2; 
            min-width: 300px;
        }
        .control-section { 
            flex: 1; 
            min-width: 280px;
            background: #16213e;
            padding: 15px;
            border-radius: 10px;
        }
        .video-container {
            background: #000;
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 10px;
        }
        .video-container img {
            width: 100%;
            display: block;
        }
        .video-row {
            display: flex;
            gap: 10px;
        }
        .video-row .video-container {
            flex: 1;
        }
        .param-group {
            margin-bottom: 15px;
            padding: 10px;
            background: #0f3460;
            border-radius: 8px;
        }
        .param-group h3 {
            color: #00d4ff;
            margin-bottom: 10px;
            font-size: 0.9em;
        }
        .param-row {
            display: flex;
            align-items: center;
            margin-bottom: 8px;
        }
        .param-row label {
            flex: 1;
            font-size: 0.85em;
        }
        .param-row input[type="range"] {
            flex: 2;
            margin: 0 10px;
        }
        .param-row .value {
            width: 50px;
            text-align: right;
            font-weight: bold;
            color: #00ff88;
        }
        .status-box {
            background: #0f3460;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 15px;
            text-align: center;
        }
        .status-box .error {
            font-size: 2em;
            font-weight: bold;
        }
        .status-box .error.left { color: #ff6b6b; }
        .status-box .error.right { color: #4ecdc4; }
        .status-box .error.center { color: #00ff88; }
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            margin: 5px;
            transition: all 0.3s;
        }
        .btn-save { background: #00ff88; color: #000; }
        .btn-load { background: #00d4ff; color: #000; }
        .btn-reset { background: #ff6b6b; color: #fff; }
        .btn-capture { background: #ffd93d; color: #000; }
        .btn:hover { transform: scale(1.05); }
        .btn-group { text-align: center; margin-top: 10px; }
        .info { font-size: 0.8em; color: #888; margin-top: 5px; }
    </style>
</head>
<body>
    <h1>🚗 Lane Detection Tuning Tool</h1>
    
    <div class="container">
        <div class="video-section">
            <div class="video-container">
                <img src="/video_feed" alt="Lane Detection">
            </div>
            <div class="video-row">
                <div class="video-container">
                    <img src="/edges_feed" alt="Canny Edges">
                </div>
            </div>
        </div>
        
        <div class="control-section">
            <div class="status-box">
                <div>Error (Sai số)</div>
                <div class="error" id="error-display">0 px</div>
                <div class="info">← Trái | Phải →</div>
            </div>
            
            <div class="param-group">
                <h3>📐 ROI (Vùng quan tâm)</h3>
                <div class="param-row">
                    <label>Top Ratio:</label>
                    <input type="range" id="roi_top_ratio" min="0.1" max="0.6" step="0.05" value="{{ config.roi_top_ratio }}">
                    <span class="value" id="roi_top_ratio_val">{{ config.roi_top_ratio }}</span>
                </div>
            </div>
            
            <div class="param-group">
                <h3>🔍 Canny Edge Detection</h3>
                <div class="param-row">
                    <label>Low:</label>
                    <input type="range" id="canny_low" min="10" max="100" step="5" value="{{ config.canny_low }}">
                    <span class="value" id="canny_low_val">{{ config.canny_low }}</span>
                </div>
                <div class="param-row">
                    <label>High:</label>
                    <input type="range" id="canny_high" min="50" max="200" step="5" value="{{ config.canny_high }}">
                    <span class="value" id="canny_high_val">{{ config.canny_high }}</span>
                </div>
            </div>
            
            <div class="param-group">
                <h3>📏 Hough Transform</h3>
                <div class="param-row">
                    <label>Threshold:</label>
                    <input type="range" id="hough_threshold" min="5" max="50" step="1" value="{{ config.hough_threshold }}">
                    <span class="value" id="hough_threshold_val">{{ config.hough_threshold }}</span>
                </div>
                <div class="param-row">
                    <label>Min Length:</label>
                    <input type="range" id="min_line_length" min="10" max="80" step="5" value="{{ config.min_line_length }}">
                    <span class="value" id="min_line_length_val">{{ config.min_line_length }}</span>
                </div>
                <div class="param-row">
                    <label>Max Gap:</label>
                    <input type="range" id="max_line_gap" min="5" max="50" step="5" value="{{ config.max_line_gap }}">
                    <span class="value" id="max_line_gap_val">{{ config.max_line_gap }}</span>
                </div>
            </div>
            
            <div class="param-group">
                <h3>🌫️ Blur</h3>
                <div class="param-row">
                    <label>Kernel:</label>
                    <input type="range" id="blur_kernel" min="3" max="15" step="2" value="{{ config.blur_kernel }}">
                    <span class="value" id="blur_kernel_val">{{ config.blur_kernel }}</span>
                </div>
            </div>
            
            <div class="btn-group">
                <button class="btn btn-save" onclick="saveConfig()">💾 Save</button>
                <button class="btn btn-load" onclick="loadConfig()">📂 Load</button>
                <button class="btn btn-reset" onclick="resetConfig()">🔄 Reset</button>
                <button class="btn btn-capture" onclick="capture()">📸 Capture</button>
            </div>
            
            <div class="info" style="margin-top: 15px; text-align: center;">
                Frame: <span id="frame-count">0</span> | 
                Config: <span id="config-status">Loaded</span>
            </div>
        </div>
    </div>
    
    <script>
        // Danh sách các parameter
        const params = ['roi_top_ratio', 'canny_low', 'canny_high', 
                       'hough_threshold', 'min_line_length', 'max_line_gap', 'blur_kernel'];
        
        // Gắn event listener cho tất cả sliders
        params.forEach(param => {
            const slider = document.getElementById(param);
            const valueSpan = document.getElementById(param + '_val');
            
            slider.addEventListener('input', function() {
                valueSpan.textContent = this.value;
                updateParam(param, this.value);
            });
        });
        
        // Cập nhật parameter lên server
        function updateParam(param, value) {
            fetch('/update_param', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({param: param, value: parseFloat(value)})
            });
        }
        
        // Lưu config
        function saveConfig() {
            fetch('/save_config', {method: 'POST'})
                .then(r => r.json())
                .then(d => {
                    document.getElementById('config-status').textContent = 'Saved!';
                    setTimeout(() => document.getElementById('config-status').textContent = 'Loaded', 2000);
                });
        }
        
        // Load config
        function loadConfig() {
            fetch('/load_config')
                .then(r => r.json())
                .then(config => {
                    params.forEach(param => {
                        if (config[param] !== undefined) {
                            document.getElementById(param).value = config[param];
                            document.getElementById(param + '_val').textContent = config[param];
                        }
                    });
                    document.getElementById('config-status').textContent = 'Loaded!';
                });
        }
        
        // Reset config
        function resetConfig() {
            fetch('/reset_config', {method: 'POST'})
                .then(r => r.json())
                .then(config => {
                    params.forEach(param => {
                        if (config[param] !== undefined) {
                            document.getElementById(param).value = config[param];
                            document.getElementById(param + '_val').textContent = config[param];
                        }
                    });
                    document.getElementById('config-status').textContent = 'Reset!';
                });
        }
        
        // Capture ảnh
        function capture() {
            fetch('/capture', {method: 'POST'})
                .then(r => r.json())
                .then(d => alert('Captured: ' + d.filename));
        }
        
        // Cập nhật status
        function updateStatus() {
            fetch('/status')
                .then(r => r.json())
                .then(d => {
                    const errorEl = document.getElementById('error-display');
                    errorEl.textContent = d.error + ' px';
                    errorEl.className = 'error ' + (d.error > 10 ? 'left' : d.error < -10 ? 'right' : 'center');
                    document.getElementById('frame-count').textContent = d.frame_count;
                });
        }
        
        // Cập nhật status mỗi 200ms
        setInterval(updateStatus, 200);
    </script>
</body>
</html>
'''


# ===== FLASK ROUTES =====

@app.route('/')
def index():
    """Trang chủ"""
    return render_template_string(HTML_TEMPLATE, config=current_config)


@app.route('/video_feed')
def video_feed():
    """Video stream với lane detection"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/edges_feed')
def edges_feed():
    """Video stream của Canny edges"""
    return Response(generate_edges(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/update_param', methods=['POST'])
def update_param():
    """Cập nhật một parameter"""
    global current_config
    data = request.json
    param = data.get('param')
    value = data.get('value')
    
    if param in current_config:
        # Chuyển đổi kiểu dữ liệu
        if param in ['roi_top_ratio', 'roi_bottom_ratio']:
            current_config[param] = float(value)
        else:
            current_config[param] = int(value)
        logger.info(f"Updated {param} = {current_config[param]}")
        return jsonify({'success': True, 'param': param, 'value': current_config[param]})
    
    return jsonify({'success': False, 'error': 'Unknown parameter'})


@app.route('/save_config', methods=['POST'])
def save_config_route():
    """Lưu config ra file"""
    success = save_config()
    return jsonify({'success': success})


@app.route('/load_config')
def load_config_route():
    """Load config từ file"""
    load_config()
    return jsonify(current_config)


@app.route('/reset_config', methods=['POST'])
def reset_config():
    """Reset về config mặc định"""
    global current_config
    current_config = DEFAULT_CONFIG.copy()
    logger.info("Config reset to defaults")
    return jsonify(current_config)


@app.route('/get_config')
def get_config():
    """Lấy config hiện tại"""
    return jsonify(current_config)


@app.route('/status')
def status():
    """Lấy trạng thái hiện tại"""
    return jsonify({
        'error': latest_error,
        'x_line': latest_x_line,
        'frame_count': frame_count,
        'config': current_config
    })


@app.route('/capture', methods=['POST'])
def capture():
    """Chụp và lưu ảnh"""
    global camera
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if camera is None:
        return jsonify({'success': False, 'error': 'Camera not initialized'})
    
    try:
        frame = camera.capture_array()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Lưu ảnh gốc
        original_path = f"{OUTPUT_DIR}/capture_{timestamp}_original.jpg"
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(original_path, frame_bgr)
        
        # Lưu ảnh debug
        _, _, _, debug_frame, edges = detect_lane(frame, current_config)
        debug_path = f"{OUTPUT_DIR}/capture_{timestamp}_debug.jpg"
        debug_bgr = cv2.cvtColor(debug_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(debug_path, debug_bgr)
        
        # Lưu edges
        edges_path = f"{OUTPUT_DIR}/capture_{timestamp}_edges.jpg"
        cv2.imwrite(edges_path, edges)
        
        logger.info(f"📸 Captured: {timestamp}")
        
        return jsonify({
            'success': True,
            'filename': f"capture_{timestamp}",
            'files': [original_path, debug_path, edges_path]
        })
        
    except Exception as e:
        logger.error(f"Capture error: {e}")
        return jsonify({'success': False, 'error': str(e)})


# ===== MAIN =====
def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("🚗 LANE DETECTION TUNING TOOL - Flask Web Version")
    print("="*60)
    
    # Tạo thư mục output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load config
    load_config()
    
    # Khởi tạo camera
    if not init_camera():
        print("❌ Failed to initialize camera!")
        print("   Make sure Picamera2 is properly installed and camera is connected.")
        return
    
    print(f"\n📁 Output directory: {OUTPUT_DIR}/")
    print(f"📄 Config file: {CONFIG_FILE}")
    print("\n🌐 Starting web server...")
    print("   Open browser: http://<raspberry_pi_ip>:5000")
    print("   Press Ctrl+C to stop\n")
    print("="*60 + "\n")
    
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
    finally:
        if camera:
            camera.stop()
            print("📷 Camera stopped")


if __name__ == "__main__":
    main()
