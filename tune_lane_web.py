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
from pathlib import Path
from flask import Flask, render_template_string, Response, request, jsonify

# Thêm đường dẫn để import module dự án
sys.path.append(str(Path(__file__).parent.parent))
from perception.camera_manager import get_web_camera

# Khởi tạo Flask App
app = Flask(__name__)

# Biến toàn cục
config_path = 'config/hardware_config.yaml'
camera = None
lock = threading.Lock()

# Load config ban đầu
def load_config():
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

full_config = load_config()
lane_cfg = full_config.get('lane_following', {}).get('lane_detection', {})

# Giá trị mặc định nếu config lỗi
current_params = {
    'roi_top_ratio': lane_cfg.get('roi_top_ratio', 0.25),
    'roi_bottom_ratio': lane_cfg.get('roi_bottom_ratio', 1.0),
    'roi_width_top': 30,  # Phần trăm thu hẹp cạnh trên (0-50%)
    'canny_low': lane_cfg.get('canny_low', 50),
    'canny_high': lane_cfg.get('canny_high', 150),
    'hough_threshold': lane_cfg.get('hough_threshold', 40),
    'blur_kernel': 7
}

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
    global current_params
    
    # 1. Resize cho nhẹ
    frame = cv2.resize(frame, (640, 480))
    height, width = frame.shape[:2]
    
    # 2. Xử lý ảnh (Canny)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (current_params['blur_kernel'], current_params['blur_kernel']), 0)
    edges = cv2.Canny(blur, current_params['canny_low'], current_params['canny_high'])
    
    # 3. Tạo ROI (Hình thang)
    roi_top = int(height * current_params['roi_top_ratio'])
    roi_bot = int(height * current_params['roi_bottom_ratio'])
    w_margin = int(width * (current_params['roi_width_top'] / 100.0))
    
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
                           int(current_params['hough_threshold']), 
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
        frame = camera.capture_frame()
        if frame is None:
            time.sleep(0.01)
            continue
            
        # Chuyển BGR để xử lý
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        processed_frame = process_frame(frame_bgr)
        
        # Encode JPEG
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

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
    data = request.json
    for key, value in data.items():
        if key in current_params:
            current_params[key] = value
    return jsonify(success=True)

@app.route('/save_config', methods=['POST'])
def save_config():
    try:
        # Load lại file gốc để tránh mất các config khác
        with open(config_path, 'r') as f:
            full_cfg = yaml.safe_load(f)
            
        # Cập nhật section lane_detection
        lane_section = full_cfg.setdefault('lane_following', {}).setdefault('lane_detection', {})
        lane_section['canny_low'] = int(current_params['canny_low'])
        lane_section['canny_high'] = int(current_params['canny_high'])
        lane_section['roi_top_ratio'] = float(current_params['roi_top_ratio'])
        lane_section['hough_threshold'] = int(current_params['hough_threshold'])
        # (Lưu ý: roi_width_top chưa có trong config gốc nên có thể chưa lưu hoặc cần thêm key mới)
        
        with open(config_path, 'w') as f:
            yaml.dump(full_cfg, f, default_flow_style=False)
            
        print("✅ Config saved successfully!")
        return jsonify(success=True)
    except Exception as e:
        print(f"❌ Error saving config: {e}")
        return jsonify(success=False)

# ===== MAIN =====
if __name__ == '__main__':
    # Init Camera từ Project
    print("🎥 Initializing CameraManager...")
    config = load_config()
    camera = get_web_camera(config)
    
    if not camera.is_running():
        camera.start()
        
    print(f"🚀 Web Tuning Tool started at http://0.0.0.0:5000")
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        if camera:
            camera.stop()