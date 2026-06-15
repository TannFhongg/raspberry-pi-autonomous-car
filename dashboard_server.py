"""
Web Dashboard Server for Lane Detection Visualization
✅ Real-time parameter tuning for lane detection
✅ Lightweight Flask server to display lane detection in browser
✅ No cv2.imshow() needed - Access via browser on any device
"""

from flask import Flask, Response, render_template_string, jsonify, request
import cv2
import numpy as np
import threading
import time
from pathlib import Path

# Import lane detector
try:
    from perception.lane_detector import detect_line
    from perception.camera_manager import crop_yuv420_frame
    from utils.config_loader import load_config
    from picamera2 import Picamera2
    CAMERA_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Import warning: {e}")
    CAMERA_AVAILABLE = False

app = Flask(__name__)

CONFIG_PATH = Path(__file__).resolve().parent / "config" / "hardware_config.yaml"

# Global variables
current_frame = None
current_debug_frame = None
current_error = 0
current_status = "STOPPED"
lane_status = "NO_LANE"
frame_lock = threading.Lock()

# ===== LANE DETECTION PARAMETERS (Tunable) =====
DEFAULT_LANE_PARAMS = {
    'roi_top_ratio': 0.6,
    'roi_bottom_ratio': 1.0,
    'roi_left_ratio': 0.10,
    'roi_right_ratio': 0.90,
    'canny_low': 80,
    'canny_high': 185,
    'hough_threshold': 40,
    'min_line_length': 50,
    'max_line_gap': 25,
    'blur_kernel': 7,
    'camera_offset': -15,
}


def parse_size(value, default=None, name="size"):
    """Parse a [width, height] config value."""
    if value is None:
        return default

    try:
        width, height = value
        width = int(width)
        height = int(height)
    except (TypeError, ValueError):
        print(f"⚠️  {name} không hợp lệ: {value!r}, dùng {default!r}")
        return default

    if width <= 0 or height <= 0:
        print(f"⚠️  {name} không hợp lệ: {value!r}, dùng {default!r}")
        return default

    return (width, height)


def load_hardware_config():
    """Load the shared robot hardware config once for dashboard startup."""
    try:
        return load_config(str(CONFIG_PATH))
    except Exception as e:
        print(f"⚠️  Không load được hardware_config.yaml, dùng mặc định dashboard: {e}")
        return {}


hardware_config = load_hardware_config()


def load_default_lane_params():
    """Load lane defaults from the main robot config, with safe fallbacks."""
    params = DEFAULT_LANE_PARAMS.copy()
    params.update(hardware_config.get('ai', {}).get('lane_detection', {}))
    return params


def load_camera_settings():
    """Load Picamera2 settings from hardware_config.yaml."""
    camera_config = hardware_config.get('sensors', {}).get('camera', {})
    picam_config = camera_config.get('picamera2', {})

    resolution = parse_size(
        camera_config.get('resolution'), (960, 720), 'camera resolution'
    )
    sensor_output_size = parse_size(
        picam_config.get('sensor_output_size', camera_config.get('sensor_output_size')),
        None,
        'camera sensor_output_size',
    )

    try:
        buffer_count = int(picam_config.get('buffer_count', 4))
    except (TypeError, ValueError):
        buffer_count = 4

    sensor_config = {}
    if sensor_output_size:
        try:
            sensor_bit_depth = int(
                picam_config.get(
                    'sensor_bit_depth',
                    camera_config.get('sensor_bit_depth', 10),
                )
            )
        except (TypeError, ValueError):
            sensor_bit_depth = 10

        sensor_config = {
            'output_size': sensor_output_size,
            'bit_depth': sensor_bit_depth,
        }

    return {
        'resolution': resolution,
        'format': picam_config.get('format', 'YUV420'),
        'buffer_count': max(1, buffer_count),
        'sensor': sensor_config,
    }


# Parameter ranges for UI sliders
PARAM_RANGES = {
    'roi_top_ratio': {'min': 0.0, 'max': 0.8, 'step': 0.05},
    'roi_bottom_ratio': {'min': 0.5, 'max': 1.0, 'step': 0.05},
    'roi_left_ratio': {'min': 0.0, 'max': 0.4, 'step': 0.05},
    'roi_right_ratio': {'min': 0.6, 'max': 1.0, 'step': 0.05},
    'canny_low': {'min': 10, 'max': 150, 'step': 5},
    'canny_high': {'min': 50, 'max': 300, 'step': 5},
    'hough_threshold': {'min': 10, 'max': 100, 'step': 5},
    'min_line_length': {'min': 10, 'max': 100, 'step': 5},
    'max_line_gap': {'min': 5, 'max': 80, 'step': 5},
    'blur_kernel': {'min': 1, 'max': 15, 'step': 2},
    'camera_offset': {'min': -200, 'max': 200, 'step': 1},
}
TUNABLE_PARAM_NAMES = list(PARAM_RANGES.keys())

lane_params = load_default_lane_params()


def validate_lane_param(name, value):
    """Validate and normalize one lane tuning parameter."""
    if name not in PARAM_RANGES:
        raise ValueError("Invalid parameter")

    if isinstance(value, bool):
        raise ValueError("Parameter value must be numeric")

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        raise ValueError("Parameter value must be numeric")

    if not np.isfinite(numeric_value):
        raise ValueError("Parameter value must be finite")

    param_range = PARAM_RANGES[name]
    numeric_value = max(param_range['min'], min(param_range['max'], numeric_value))

    if name.startswith('roi_'):
        return float(numeric_value)

    int_value = int(round(numeric_value))

    if name == 'blur_kernel':
        int_value = max(param_range['min'], min(param_range['max'], int_value))
        if int_value % 2 == 0:
            int_value += 1
        if int_value > param_range['max']:
            int_value -= 2
        return int_value

    return max(param_range['min'], min(param_range['max'], int_value))


def format_yaml_scalar(value):
    """Format dashboard numeric values for YAML."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def save_tunable_lane_params_to_config(config_path=None):
    """Persist current slider values into ai.lane_detection while preserving comments."""
    path = Path(config_path or CONFIG_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    values = {
        name: lane_params[name]
        for name in TUNABLE_PARAM_NAMES
        if name in lane_params
    }
    lines = path.read_text(encoding='utf-8').splitlines()

    ai_start = None
    for index, line in enumerate(lines):
        if line.strip().startswith('#') or not line.strip():
            continue
        if line.startswith('ai:'):
            ai_start = index
            break

    if ai_start is None:
        lines.extend(['', 'ai:', '  lane_detection:'])
        ai_start = len(lines) - 2

    ai_end = len(lines)
    for index in range(ai_start + 1, len(lines)):
        line = lines[index]
        if line.strip() and not line.startswith(' ') and not line.strip().startswith('#'):
            ai_end = index
            break

    lane_start = None
    for index in range(ai_start + 1, ai_end):
        stripped = lines[index].strip()
        if lines[index].startswith('  ') and not lines[index].startswith('    ') and stripped.startswith('lane_detection:'):
            lane_start = index
            break

    if lane_start is None:
        lines.insert(ai_end, '  lane_detection:')
        lane_start = ai_end
        ai_end += 1

    lane_end = ai_end
    for index in range(lane_start + 1, ai_end):
        line = lines[index]
        if line.strip() and not line.startswith('    ') and not line.strip().startswith('#'):
            lane_end = index
            break

    updated = set()
    for index in range(lane_start + 1, lane_end):
        line = lines[index]
        stripped = line.strip()
        if not stripped or stripped.startswith('#') or ':' not in stripped:
            continue

        key = stripped.split(':', 1)[0].strip()
        if key not in values:
            continue

        before_comment, separator, comment = line.partition('#')
        prefix = before_comment.split(':', 1)[0] + ': '
        comment_padding = before_comment[len(before_comment.rstrip()):]
        suffix = f"{comment_padding}#{comment}" if separator else ''
        lines[index] = prefix + format_yaml_scalar(values[key]) + suffix
        updated.add(key)

    missing = [name for name in TUNABLE_PARAM_NAMES if name in values and name not in updated]
    insert_lines = [f"    {name}: {format_yaml_scalar(values[name])}" for name in missing]
    if insert_lines:
        lines[lane_end:lane_end] = insert_lines

    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

# HTML Template with embedded CSS/JS
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚗 Lane Detection Tuning Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff;
            min-height: 100vh;
            padding: 15px;
        }
        
        .container { max-width: 1600px; margin: 0 auto; }
        
        header { text-align: center; margin-bottom: 20px; }
        header h1 { font-size: 2em; text-shadow: 2px 2px 4px rgba(0,0,0,0.5); }
        header p { opacity: 0.7; margin-top: 5px; }
        
        .dashboard {
            display: grid;
            grid-template-columns: 1fr 400px;
            gap: 15px;
        }
        
        @media (max-width: 1200px) {
            .dashboard { grid-template-columns: 1fr; }
        }
        
        .panel {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 15px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .panel h2 {
            margin-bottom: 12px;
            font-size: 1.2em;
            color: #4fc3f7;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            padding-bottom: 8px;
        }
        
        #camera-feed {
            width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.4);
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin-top: 12px;
        }
        
        .stat-card {
            background: rgba(255,255,255,0.08);
            padding: 12px;
            border-radius: 8px;
            text-align: center;
        }
        
        .stat-card label {
            display: block;
            font-size: 0.75em;
            opacity: 0.7;
            margin-bottom: 4px;
        }
        
        .stat-card .value {
            font-size: 1.5em;
            font-weight: bold;
        }
        
        .error-positive { color: #ff6b6b; }
        .error-negative { color: #51cf66; }
        .error-zero { color: #ffd43b; }
        .status-running { color: #51cf66; }
        .status-stopped { color: #ff6b6b; }
        
        /* Parameter Tuning Panel */
        .param-group {
            margin-bottom: 12px;
            padding: 10px;
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
        }
        
        .param-group label {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
            font-size: 0.85em;
        }
        
        .param-group label span.param-name { color: #4fc3f7; }
        .param-group label span.param-value {
            background: #4fc3f7;
            color: #1a1a2e;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: bold;
            min-width: 50px;
            text-align: center;
        }
        
        .param-group input[type="range"] {
            width: 100%;
            height: 6px;
            border-radius: 3px;
            background: rgba(255,255,255,0.2);
            outline: none;
            -webkit-appearance: none;
        }
        
        .param-group input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: #4fc3f7;
            cursor: pointer;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        }
        
        .controls {
            display: flex;
            gap: 8px;
            margin-top: 15px;
            flex-wrap: wrap;
        }
        
        .btn {
            padding: 10px 16px;
            border: none;
            border-radius: 6px;
            font-size: 0.9em;
            cursor: pointer;
            transition: all 0.2s;
            font-weight: 600;
            flex: 1;
            min-width: 100px;
        }
        
        .btn-primary { background: #51cf66; color: white; }
        .btn-danger { background: #ff6b6b; color: white; }
        .btn-warning { background: #ffd43b; color: #1a1a2e; }
        .btn-info { background: #4fc3f7; color: #1a1a2e; }
        
        .btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        
        .fps-counter {
            position: fixed;
            top: 15px;
            right: 15px;
            background: rgba(0,0,0,0.7);
            padding: 8px 15px;
            border-radius: 6px;
            font-size: 1em;
            font-weight: bold;
            z-index: 100;
        }
        
        .section-title {
            color: #ffd43b;
            font-size: 0.9em;
            margin: 15px 0 8px 0;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .toast {
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: #51cf66;
            color: white;
            padding: 12px 24px;
            border-radius: 8px;
            font-weight: bold;
            opacity: 0;
            transition: opacity 0.3s;
            z-index: 1000;
        }
        
        .toast.show { opacity: 1; }
        .toast.error { background: #ff6b6b; }
    </style>
</head>
<body>
    <div class="fps-counter">FPS: <span id="fps">0</span></div>
    <div class="toast" id="toast"></div>
    
    <div class="container">
        <header>
            <h1>🚗 Lane Detection Tuning Dashboard</h1>
            <p>Real-time Parameter Adjustment</p>
        </header>
        
        <div class="dashboard">
            <!-- Left: Video Feed + Stats -->
            <div>
                <div class="panel">
                    <h2>📹 Camera Feed (Debug View)</h2>
                    <img id="camera-feed" src="/video_feed" alt="Camera Feed">
                    
                    <div class="stats">
                        <div class="stat-card">
                            <label>Error</label>
                            <div class="value" id="error">0</div>
                        </div>
                        <div class="stat-card">
                            <label>Lane Status</label>
                            <div class="value" id="lane-status" style="font-size:1em;">NO_LANE</div>
                        </div>
                        <div class="stat-card">
                            <label>Robot Status</label>
                            <div class="value" id="robot-status" style="font-size:1em;">STOPPED</div>
                        </div>
                        <div class="stat-card">
                            <label>Speed</label>
                            <div class="value" id="speed">0</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Right: Parameter Tuning -->
            <div class="panel">
                <h2>⚙️ Lane Detection Parameters</h2>
                
                <div class="section-title">📐 ROI (Region of Interest)</div>
                
                <div class="param-group">
                    <label>
                        <span class="param-name">roi_top_ratio</span>
                        <span class="param-value" id="val-roi_top_ratio">{{ lane_params.roi_top_ratio }}</span>
                    </label>
                    <input type="range" id="roi_top_ratio" min="0" max="0.8" step="0.05" value="{{ lane_params.roi_top_ratio }}">
                </div>
                
                <div class="param-group">
                    <label>
                        <span class="param-name">roi_bottom_ratio</span>
                        <span class="param-value" id="val-roi_bottom_ratio">{{ lane_params.roi_bottom_ratio }}</span>
                    </label>
                    <input type="range" id="roi_bottom_ratio" min="0.5" max="1.0" step="0.05" value="{{ lane_params.roi_bottom_ratio }}">
                </div>
                
                <div class="param-group">
                    <label>
                        <span class="param-name">roi_left_ratio</span>
                        <span class="param-value" id="val-roi_left_ratio">{{ lane_params.roi_left_ratio }}</span>
                    </label>
                    <input type="range" id="roi_left_ratio" min="0.0" max="0.4" step="0.05" value="{{ lane_params.roi_left_ratio }}">
                </div>
                
                <div class="param-group">
                    <label>
                        <span class="param-name">roi_right_ratio</span>
                        <span class="param-value" id="val-roi_right_ratio">{{ lane_params.roi_right_ratio }}</span>
                    </label>
                    <input type="range" id="roi_right_ratio" min="0.6" max="1.0" step="0.05" value="{{ lane_params.roi_right_ratio }}">
                </div>
                
                <div class="section-title">🔍 Canny Edge Detection</div>
                
                <div class="param-group">
                    <label>
                        <span class="param-name">canny_low</span>
                        <span class="param-value" id="val-canny_low">{{ lane_params.canny_low }}</span>
                    </label>
                    <input type="range" id="canny_low" min="10" max="150" step="5" value="{{ lane_params.canny_low }}">
                </div>
                
                <div class="param-group">
                    <label>
                        <span class="param-name">canny_high</span>
                        <span class="param-value" id="val-canny_high">{{ lane_params.canny_high }}</span>
                    </label>
                    <input type="range" id="canny_high" min="50" max="300" step="5" value="{{ lane_params.canny_high }}">
                </div>
                
                <div class="section-title">📏 Hough Transform</div>
                
                <div class="param-group">
                    <label>
                        <span class="param-name">hough_threshold</span>
                        <span class="param-value" id="val-hough_threshold">{{ lane_params.hough_threshold }}</span>
                    </label>
                    <input type="range" id="hough_threshold" min="10" max="100" step="5" value="{{ lane_params.hough_threshold }}">
                </div>
                
                <div class="param-group">
                    <label>
                        <span class="param-name">min_line_length</span>
                        <span class="param-value" id="val-min_line_length">{{ lane_params.min_line_length }}</span>
                    </label>
                    <input type="range" id="min_line_length" min="10" max="100" step="5" value="{{ lane_params.min_line_length }}">
                </div>
                
                <div class="param-group">
                    <label>
                        <span class="param-name">max_line_gap</span>
                        <span class="param-value" id="val-max_line_gap">{{ lane_params.max_line_gap }}</span>
                    </label>
                    <input type="range" id="max_line_gap" min="5" max="80" step="5" value="{{ lane_params.max_line_gap }}">
                </div>
                
                <div class="section-title">🌫️ Preprocessing</div>
                
                <div class="param-group">
                    <label>
                        <span class="param-name">blur_kernel</span>
                        <span class="param-value" id="val-blur_kernel">{{ lane_params.blur_kernel }}</span>
                    </label>
                    <input type="range" id="blur_kernel" min="1" max="15" step="2" value="{{ lane_params.blur_kernel }}">
                </div>

                <div class="section-title">🎯 Calibration Offset</div>

                <div class="param-group">
                    <label>
                        <span class="param-name">camera_offset</span>
                        <span class="param-value" id="val-camera_offset">{{ lane_params.camera_offset }}</span>
                    </label>
                    <input type="range" id="camera_offset" min="-200" max="200" step="1" value="{{ lane_params.camera_offset }}">
                </div>
                
                <div class="controls">
                    <button class="btn btn-warning" onclick="resetParams()">🔄 Reset</button>
                    <button class="btn btn-info" onclick="copyParams()">📋 Copy</button>
                    <button class="btn btn-primary" onclick="saveParams()">💾 Save</button>
                </div>
                
                <div class="controls" style="margin-top: 10px;">
                    <button class="btn btn-primary" onclick="startRobot()">▶️ Start</button>
                    <button class="btn btn-danger" onclick="stopRobot()">⏹️ Stop</button>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Default parameters
        const defaultParams = {{ lane_params | tojson }};
        const tunableParamNames = {{ tunable_param_names | tojson }};
        
        // Initialize sliders
        const paramNames = tunableParamNames;
        
        paramNames.forEach(name => {
            const slider = document.getElementById(name);
            const valueDisplay = document.getElementById('val-' + name);
            if (!slider || !valueDisplay) {
                console.warn('Missing slider element for parameter:', name);
                return;
            }
            
            slider.addEventListener('input', () => {
                const val = parseFloat(slider.value);
                valueDisplay.textContent = val;
                updateParam(name, val);
            });
        });
        
        // Update parameter on server
        async function updateParam(name, value) {
            try {
                const response = await fetch('/update_param', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name, value })
                });
                const data = await response.json();
                if (!response.ok || data.status !== 'ok') {
                    throw new Error(data.message || 'Update failed');
                }

                const slider = document.getElementById(name);
                const valueDisplay = document.getElementById('val-' + name);
                if (slider && valueDisplay) {
                    slider.value = data.value;
                    valueDisplay.textContent = data.value;
                }
            } catch (e) {
                console.error('Error updating param:', e);
                showToast('Không cập nhật được ' + name + ': ' + e.message, true);
            }
        }
        
        // Reset to defaults
        function resetParams() {
            paramNames.forEach(name => {
                const slider = document.getElementById(name);
                const valueDisplay = document.getElementById('val-' + name);
                slider.value = defaultParams[name];
                valueDisplay.textContent = defaultParams[name];
                updateParam(name, defaultParams[name]);
            });
            showToast('Parameters reset to defaults');
        }
        
        // Copy params as Python dict
        function copyParams() {
            const params = {};
            paramNames.forEach(name => {
                params[name] = parseFloat(document.getElementById(name).value);
            });
            
            let text = "lane_params = {\\n";
            for (const [key, val] of Object.entries(params)) {
                text += `    '${key}': ${val},\\n`;
            }
            text += "}";
            
            navigator.clipboard.writeText(text).then(() => {
                showToast('Copied to clipboard!');
            });
        }
        
        // Save params to server
        async function saveParams() {
            try {
                const response = await fetch('/save_params', { method: 'POST' });
                const data = await response.json();
                showToast(data.message || 'Parameters saved!');
            } catch (e) {
                showToast('Error saving params', true);
            }
        }
        
        // Toast notification
        function showToast(message, isError = false) {
            const toast = document.getElementById('toast');
            toast.textContent = message;
            toast.className = 'toast show' + (isError ? ' error' : '');
            setTimeout(() => toast.className = 'toast', 2000);
        }
        
        // FPS counter
        let frameCount = 0;
        let lastTime = Date.now();
        
        setInterval(() => {
            const now = Date.now();
            const fps = Math.round(frameCount * 1000 / (now - lastTime));
            document.getElementById('fps').textContent = fps;
            frameCount = 0;
            lastTime = now;
        }, 1000);
        
        document.getElementById('camera-feed').onload = () => frameCount++;
        
        // Update stats
        setInterval(async () => {
            try {
                const response = await fetch('/stats');
                const data = await response.json();
                
                const errorEl = document.getElementById('error');
                errorEl.textContent = data.error > 0 ? '+' + data.error : data.error;
                errorEl.className = 'value ' + 
                    (data.error > 20 ? 'error-positive' : 
                     data.error < -20 ? 'error-negative' : 'error-zero');
                
                document.getElementById('lane-status').textContent = data.lane_status;
                document.getElementById('robot-status').textContent = data.robot_status;
                document.getElementById('speed').textContent = data.speed;
            } catch (e) {}
        }, 100);
        
        // Load current params from server
        async function loadParams() {
            try {
                const response = await fetch('/get_params');
                const params = await response.json();
                
                paramNames.forEach(name => {
                    if (params[name] !== undefined) {
                        const slider = document.getElementById(name);
                        const valueDisplay = document.getElementById('val-' + name);
                        if (!slider || !valueDisplay) return;
                        slider.value = params[name];
                        valueDisplay.textContent = params[name];
                    }
                });
            } catch (e) {
                console.error('Error loading params:', e);
            }
        }
        
        // Control functions
        function startRobot() {
            fetch('/start', { method: 'POST' });
            showToast('Robot started');
        }
        
        function stopRobot() {
            fetch('/stop', { method: 'POST' });
            showToast('Robot stopped');
        }
        
        // Load params on page load
        loadParams();
    </script>
</body>
</html>
"""


def classify_lane_status(error):
    """Return a dashboard-friendly status from the detector error value."""
    if error == 999:
        return "NO_LANE"
    if error > 20:
        return "OFFSET_RIGHT"
    if error < -20:
        return "OFFSET_LEFT"
    return "CENTERED"


def _dashboard_gray_frame(frame):
    """Extract grayscale for dashboard-only line debugging."""
    if frame is None or not hasattr(frame, "shape"):
        return None

    if len(frame.shape) == 2:
        height, width = frame.shape[:2]
        if height % 3 == 0 and width % 2 == 0:
            image_height = (height * 2) // 3
            return frame[:image_height, :].copy()
        return frame.copy()

    if len(frame.shape) == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return None


def _odd_kernel(value, default=7):
    try:
        kernel = int(round(float(value)))
    except (TypeError, ValueError):
        kernel = default
    kernel = max(1, kernel)
    if kernel % 2 == 0:
        kernel += 1
    return kernel


def _calculate_lane_x_at_bottom(lines, height, width):
    x_bottoms = []
    for x1, y1, x2, y2, slope in lines:
        x_bottom = x1 + (height - y1) / slope
        if 0 <= x_bottom <= width:
            x_bottoms.append(x_bottom)

    if not x_bottoms:
        return None
    return int(np.median(x_bottoms))


def draw_detected_hough_lines(output, source_frame, params):
    """Draw dashboard-only raw Hough line segments for lane tuning."""
    gray = _dashboard_gray_frame(source_frame)
    if gray is None:
        return output

    height, width = gray.shape[:2]
    center_x = width // 2

    gray_inverted = cv2.bitwise_not(gray)
    blur_kernel = _odd_kernel(params.get('blur_kernel', 7))
    blur = cv2.GaussianBlur(gray_inverted, (blur_kernel, blur_kernel), 0)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(blur)
    edges = cv2.Canny(
        enhanced,
        int(params.get('canny_low', 80)),
        int(params.get('canny_high', 185)),
    )

    roi_top = int(height * params.get('roi_top_ratio', 0.6))
    roi_bottom = int(height * params.get('roi_bottom_ratio', 1.0))
    roi_left = int(width * params.get('roi_left_ratio', 0.1))
    roi_right = int(width * params.get('roi_right_ratio', 0.9))
    roi_vertices = np.array([[
        (0, roi_bottom),
        (roi_left, roi_top),
        (roi_right, roi_top),
        (width, roi_bottom),
    ]], dtype=np.int32)

    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=int(params.get('hough_threshold', 40)),
        minLineLength=int(params.get('min_line_length', 50)),
        maxLineGap=int(params.get('max_line_gap', 25)),
    )

    left_lines = []
    right_lines = []
    rejected_count = 0

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) < 1:
                rejected_count += 1
                continue

            slope = (y2 - y1) / (x2 - x1)
            mid_x = (x1 + x2) / 2

            if slope < -0.5 and mid_x < center_x:
                left_lines.append((x1, y1, x2, y2, slope))
                cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 3)
            elif slope > 0.5 and mid_x > center_x:
                right_lines.append((x1, y1, x2, y2, slope))
                cv2.line(output, (x1, y1), (x2, y2), (255, 0, 0), 3)
            else:
                rejected_count += 1
                cv2.line(output, (x1, y1), (x2, y2), (80, 80, 80), 1)

    left_x = _calculate_lane_x_at_bottom(left_lines, height, width)
    right_x = _calculate_lane_x_at_bottom(right_lines, height, width)

    if left_x is not None:
        cv2.circle(output, (left_x, height - 12), 10, (0, 255, 0), -1)
        cv2.line(output, (left_x, height - 80), (left_x, height), (0, 255, 0), 2)

    if right_x is not None:
        cv2.circle(output, (right_x, height - 12), 10, (255, 0, 0), -1)
        cv2.line(output, (right_x, height - 80), (right_x, height), (255, 0, 0), 2)

    if left_x is not None and right_x is not None:
        cv2.line(output, (left_x, height - 30), (right_x, height - 30), (0, 255, 255), 3)
        cv2.putText(
            output,
            f"Lane px: {right_x - left_x}",
            (10, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )

    cv2.putText(
        output,
        f"Hough L:{len(left_lines)} R:{len(right_lines)} rejected:{rejected_count}",
        (10, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        output,
        "Green=left lane | Blue=right lane | Gray=rejected",
        (10, height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )

    return output


def draw_dashboard_overlay(frame, x_line, center_x, error, params, source_frame=None):
    """Draw ROI, center line, detected target line, and current error."""
    if frame is None:
        return None

    output = frame.copy()
    height, width = output.shape[:2]

    line_source = frame if source_frame is None else source_frame
    output = draw_detected_hough_lines(output, line_source, params)

    roi_top = int(height * params.get('roi_top_ratio', 0.6))
    roi_bottom = int(height * params.get('roi_bottom_ratio', 1.0))
    roi_left = int(width * params.get('roi_left_ratio', 0.1))
    roi_right = int(width * params.get('roi_right_ratio', 0.9))
    roi_vertices = np.array([[
        (0, roi_bottom),
        (roi_left, roi_top),
        (roi_right, roi_top),
        (width, roi_bottom),
    ]], dtype=np.int32)

    cv2.polylines(output, roi_vertices, True, (255, 0, 0), 2)
    cv2.line(output, (center_x, 0), (center_x, height), (0, 255, 255), 2)

    status = classify_lane_status(error)
    if error != 999:
        cv2.line(output, (x_line, 0), (x_line, height), (255, 0, 255), 3)
        cv2.arrowedLine(
            output,
            (center_x, height - 45),
            (x_line, height - 45),
            (0, 0, 255),
            3,
        )

    cv2.putText(
        output,
        f"Error: {error:+d}px | {status}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0) if error != 999 else (0, 0, 255),
        2,
    )
    return output


def camera_thread():
    """
    Camera capture thread - Chạy liên tục trong background
    
    ✅ FIX: Dùng YUV420 format consistent với camera_manager.py
    """
    global current_frame, current_debug_frame, current_error, lane_status, lane_params
    
    if not CAMERA_AVAILABLE:
        print("❌ Camera not available")
        return
    
    try:
        picam2 = Picamera2()
        camera_settings = load_camera_settings()
        
        # ============================================================
        # ✅ FIX: Dùng cấu hình camera chung với camera_manager.py
        # ============================================================
        # Output trung gian 960x720, sensor full FOV để không crop mất line.
        # ============================================================
        config_kwargs = {
            "main": {
                "size": camera_settings["resolution"],
                "format": camera_settings["format"],
            },
            "buffer_count": camera_settings["buffer_count"],
        }
        if camera_settings["sensor"]:
            config_kwargs["sensor"] = camera_settings["sensor"]

        config = picam2.create_preview_configuration(**config_kwargs)
        picam2.configure(config)
        picam2.start()
        
        width, height = camera_settings["resolution"]
        print(f"✅ Camera started ({width}x{height}, {camera_settings['format']})")
        if camera_settings["sensor"]:
            print(f"   Sensor: {camera_settings['sensor']}")
        time.sleep(2)  # Warm-up
        
        while True:
            # Capture frame (YUV420 planar format)
            frame_yuv = picam2.capture_array()
            if str(camera_settings["format"]).upper() == "YUV420":
                frame_yuv = crop_yuv420_frame(frame_yuv, camera_settings["resolution"])

            # Detect lane on raw YUV420 so the detector can use the Y channel directly.
            # debug=True asks for a BGR frame only for browser visualization.
            error, x_line, center_x, debug_frame = detect_line(
                frame_yuv, lane_params, debug=True
            )
            debug_frame = draw_dashboard_overlay(
                debug_frame, x_line, center_x, error, lane_params, frame_yuv
            )
            
            # Update global variables
            with frame_lock:
                current_frame = frame_yuv
                current_debug_frame = debug_frame
                current_error = error
                lane_status = classify_lane_status(error)
            
            time.sleep(0.03)  # ~30 FPS
            
    except Exception as e:
        print(f"❌ Camera thread error: {e}")


def generate_frames():
    """Generator function for video streaming"""
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
    return render_template_string(
        HTML_TEMPLATE,
        lane_params=lane_params,
        tunable_param_names=TUNABLE_PARAM_NAMES,
    )


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
            'speed': 0
        })


@app.route('/get_params')
def get_params():
    """Get current lane detection parameters"""
    return jsonify(lane_params)


@app.route('/update_param', methods=['POST'])
def update_param():
    """Update a single parameter"""
    global lane_params
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'status': 'error', 'message': 'Invalid JSON payload'}), 400

    name = data.get('name')
    value = data.get('value')

    try:
        value = validate_lane_param(name, value)
    except ValueError as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

    lane_params[name] = value
    print(f"📝 Updated {name} = {value}")
    return jsonify({'status': 'ok', 'name': name, 'value': value})


@app.route('/save_params', methods=['POST'])
def save_params():
    """Save current parameters to config file"""
    try:
        save_tunable_lane_params_to_config()
        print(f"💾 Parameters saved to {CONFIG_PATH}")
        return jsonify({
            'status': 'ok',
            'message': 'Saved to config/hardware_config.yaml',
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


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
    print("🌐 LANE DETECTION TUNING DASHBOARD")
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
    print("   http://localhost:5001")
    print("   hoặc http://<IP-của-Pi>:5001")
    print("\n⚙️  Tham số mặc định:")
    for k, v in lane_params.items():
        print(f"   {k}: {v}")
    print("\n⏹️  Nhấn Ctrl+C để dừng server")
    print("="*70 + "\n")
    
    # Run Flask server
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
