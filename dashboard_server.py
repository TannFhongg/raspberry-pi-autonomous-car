"""
Web Dashboard Server for Lane Detection Visualization
Lightweight Flask server to display lane detection in browser
No cv2.imshow() needed - Access via browser on any device
"""

from flask import Flask, Response, render_template_string, jsonify
import cv2
import numpy as np
import threading
import time
from io import BytesIO
import base64

# Import lane detector
try:
    from perception.lane_detector import detect_line
    from picamera2 import Picamera2
    CAMERA_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Import warning: {e}")
    CAMERA_AVAILABLE = False

app = Flask(__name__)

# Global variables
current_frame = None
current_debug_frame = None
current_error = 0
current_status = "STOPPED"
lane_status = "NO_LANE"
frame_lock = threading.Lock()

# HTML Template with embedded CSS/JS
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚗 Lane Detection Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff;
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        header h1 {
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .dashboard {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        
        @media (max-width: 768px) {
            .dashboard {
                grid-template-columns: 1fr;
            }
        }
        
        .panel {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        }
        
        .panel h2 {
            margin-bottom: 15px;
            font-size: 1.5em;
            border-bottom: 2px solid rgba(255,255,255,0.3);
            padding-bottom: 10px;
        }
        
        #camera-feed {
            width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        
        .stats {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 15px;
        }
        
        .stat-card {
            background: rgba(255,255,255,0.15);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        
        .stat-card label {
            display: block;
            font-size: 0.9em;
            opacity: 0.8;
            margin-bottom: 5px;
        }
        
        .stat-card .value {
            font-size: 2em;
            font-weight: bold;
        }
        
        .error-positive { color: #ff6b6b; }
        .error-negative { color: #51cf66; }
        .error-zero { color: #ffd43b; }
        
        .status-running { color: #51cf66; }
        .status-stopped { color: #ff6b6b; }
        
        .lane-both { color: #51cf66; }
        .lane-left { color: #ffd43b; }
        .lane-right { color: #ffd43b; }
        .lane-none { color: #ff6b6b; }
        
        .controls {
            margin-top: 20px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 1em;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: 600;
        }
        
        .btn-primary {
            background: #51cf66;
            color: white;
        }
        
        .btn-danger {
            background: #ff6b6b;
            color: white;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        
        .log {
            background: rgba(0,0,0,0.3);
            padding: 15px;
            border-radius: 10px;
            max-height: 300px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }
        
        .log-entry {
            margin: 5px 0;
            padding: 5px;
            border-left: 3px solid #51cf66;
            padding-left: 10px;
        }
        
        .fps-counter {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(0,0,0,0.7);
            padding: 10px 20px;
            border-radius: 8px;
            font-size: 1.2em;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="fps-counter">FPS: <span id="fps">0</span></div>
    
    <div class="container">
        <header>
            <h1>🚗 Lane Detection Dashboard</h1>
            <p>Real-time Autonomous Car Monitoring</p>
        </header>
        
        <div class="dashboard">
            <!-- Video Feed Panel -->
            <div class="panel">
                <h2>📹 Camera Feed</h2>
                <img id="camera-feed" src="/video_feed" alt="Camera Feed">
            </div>
            
            <!-- Statistics Panel -->
            <div class="panel">
                <h2>📊 Statistics</h2>
                <div class="stats">
                    <div class="stat-card">
                        <label>Error</label>
                        <div class="value" id="error">0</div>
                    </div>
                    
                    <div class="stat-card">
                        <label>Lane Status</label>
                        <div class="value" id="lane-status">NO_LANE</div>
                    </div>
                    
                    <div class="stat-card">
                        <label>Robot Status</label>
                        <div class="value" id="robot-status">STOPPED</div>
                    </div>
                    
                    <div class="stat-card">
                        <label>Speed</label>
                        <div class="value" id="speed">0</div>
                    </div>
                </div>
                
                <div class="controls">
                    <button class="btn btn-primary" onclick="startRobot()">▶️ Start</button>
                    <button class="btn btn-danger" onclick="stopRobot()">⏹️ Stop</button>
                </div>
            </div>
            
            <!-- Activity Log Panel -->
            <div class="panel" style="grid-column: span 2;">
                <h2>📝 Activity Log</h2>
                <div class="log" id="log">
                    <div class="log-entry">Dashboard initialized...</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let frameCount = 0;
        let lastTime = Date.now();
        
        // Update stats every 100ms
        setInterval(async () => {
            try {
                const response = await fetch('/stats');
                const data = await response.json();
                
                // Update error
                const errorEl = document.getElementById('error');
                errorEl.textContent = data.error > 0 ? '+' + data.error : data.error;
                errorEl.className = 'value ' + 
                    (data.error > 20 ? 'error-positive' : 
                     data.error < -20 ? 'error-negative' : 'error-zero');
                
                // Update lane status
                const laneEl = document.getElementById('lane-status');
                laneEl.textContent = data.lane_status;
                laneEl.className = 'value lane-' + data.lane_status.toLowerCase().split('_')[0];
                
                // Update robot status
                const statusEl = document.getElementById('robot-status');
                statusEl.textContent = data.robot_status;
                statusEl.className = 'value status-' + data.robot_status.toLowerCase();
                
                // Update speed
                document.getElementById('speed').textContent = data.speed;
                
            } catch (error) {
                console.error('Error fetching stats:', error);
            }
        }, 100);
        
        // FPS counter
        setInterval(() => {
            const now = Date.now();
            const fps = Math.round(frameCount * 1000 / (now - lastTime));
            document.getElementById('fps').textContent = fps;
            frameCount = 0;
            lastTime = now;
        }, 1000);
        
        // Track frames
        document.getElementById('camera-feed').onload = () => frameCount++;
        
        // Control functions
        function startRobot() {
            fetch('/start', { method: 'POST' });
            addLog('Robot started');
        }
        
        function stopRobot() {
            fetch('/stop', { method: 'POST' });
            addLog('Robot stopped');
        }
        
        function addLog(message) {
            const log = document.getElementById('log');
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            const time = new Date().toLocaleTimeString();
            entry.textContent = `[${time}] ${message}`;
            log.insertBefore(entry, log.firstChild);
            
            // Keep only last 50 entries
            while (log.children.length > 50) {
                log.removeChild(log.lastChild);
            }
        }
    </script>
</body>
</html>
"""


def camera_thread():
    """
    Camera capture thread - Chạy liên tục trong background
    """
    global current_frame, current_debug_frame, current_error, lane_status
    
    if not CAMERA_AVAILABLE:
        print("❌ Camera not available")
        return
    
    try:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"size": (1640, 1232), "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()
        
        print("✅ Camera started")
        time.sleep(2)  # Warm-up
        
        while True:
            # Capture frame
            frame_rgb = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Detect lane
            error, x_line, center_x, debug_frame = detect_line(frame_bgr)
            
            # Update global variables
            with frame_lock:
                current_frame = frame_bgr
                current_debug_frame = debug_frame
                current_error = error
                
                # Determine lane status
                if x_line == center_x:
                    lane_status = "NO_LANE"
                elif error > 20:
                    lane_status = "RIGHT_ONLY"
                elif error < -20:
                    lane_status = "LEFT_ONLY"
                else:
                    lane_status = "BOTH_LANES"
            
            time.sleep(0.03)  # ~30 FPS
            
    except Exception as e:
        print(f"❌ Camera thread error: {e}")


def generate_frames():
    """
    Generator function for video streaming
    """
    while True:
        with frame_lock:
            if current_debug_frame is not None:
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', current_debug_frame, 
                                          [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_bytes = buffer.tobytes()
                
                # Yield frame in multipart format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.03)  # ~30 FPS


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stats')
def stats():
    """Get current statistics as JSON"""
    with frame_lock:
        return jsonify({
            'error': current_error,
            'lane_status': lane_status,
            'robot_status': current_status,
            'speed': 0  # TODO: Integrate with motor controller
        })


@app.route('/start', methods=['POST'])
def start_robot():
    """Start robot"""
    global current_status
    current_status = "RUNNING"
    print("▶️  Robot started")
    return jsonify({'status': 'ok'})


@app.route('/stop', methods=['POST'])
def stop_robot():
    """Stop robot"""
    global current_status
    current_status = "STOPPED"
    print("⏹️  Robot stopped")
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🌐 LANE DETECTION WEB DASHBOARD")
    print("="*70)
    
    # Start camera thread
    if CAMERA_AVAILABLE:
        cam_thread = threading.Thread(target=camera_thread, daemon=True)
        cam_thread.start()
        print("📹 Camera thread started")
    else:
        print("⚠️  Running in demo mode (no camera)")
    
    print("\n🚀 Starting web server...")
    print("📱 Truy cập dashboard tại:")
    print("   http://localhost:5000")
    print("   hoặc http://<IP-của-Pi>:5000")
    print("\n💡 Để truy cập từ điện thoại/máy tính khác:")
    print("   1. Đảm bảo cùng mạng WiFi")
    print("   2. Tìm IP của Pi: hostname -I")
    print("   3. Mở browser: http://<IP-của-Pi>:5000")
    print("\n⏹️  Nhấn Ctrl+C để dừng server")
    print("="*70 + "\n")
    
    # Run Flask server
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)