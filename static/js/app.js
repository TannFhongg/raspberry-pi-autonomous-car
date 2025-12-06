// ===== SOCKET.IO CONNECTION =====
const socket = io();

// State variables
let currentMode = 'manual'; // 'manual', 'auto', 'follow'
let startTime = Date.now();
let selectedColor = 'red';
let followDistance = 50;

// Socket event handlers
socket.on('connect', function () {
    console.log('Connected to server');
    updateConnectionStatus(true);
});

socket.on('disconnect', function () {
    console.log('Disconnected from server');
    updateConnectionStatus(false);
});

// Receive sensor data from server
socket.on('sensor_update', function (data) {
    console.log('Sensor data received:', data);
    updateSensorData(data);
});

// Receive mode update from server
socket.on('mode_update', function (data) {
    console.log('Mode update received:', data);
    if (data.mode) {
        currentMode = data.mode;
        updateModeRadioButtons();
        updateControlsState();
        updateFollowStatusVisibility(); // ✅ NEW: Hide/show follow status
    }
});

// Receive target tracking data
socket.on('target_update', function (data) {
    console.log('Target data received:', data);
    updateTargetData(data);
});

// Receive log entries from server
socket.on('log_entry', function (data) {
    addLogEntry(data.time, data.level, data.message);
});

// ===== CONNECTION STATUS =====
function updateConnectionStatus(isConnected) {
    const badge = document.getElementById('connectionBadge');
    const text = document.getElementById('connectionText');

    if (isConnected) {
        badge.classList.remove('offline');
        text.textContent = 'Connected';
    } else {
        badge.classList.add('offline');
        text.textContent = 'Disconnected';
    }
}

// ===== SENSOR DATA UPDATE =====
function updateSensorData(data) {
    // Update Battery
    if (data.battery !== undefined) {
        const batteryBar = document.getElementById('batteryBar');
        const batteryValue = document.getElementById('batteryValue');
        batteryBar.style.width = data.battery + '%';
        batteryValue.textContent = data.battery + '%';
    }

    // Update Speed
    if (data.speed !== undefined) {
        const speedBar = document.getElementById('speedBar');
        const speedValue = document.getElementById('speedValue');
        const percentage = (data.speed / 255 * 100).toFixed(0);
        speedBar.style.width = percentage + '%';
        speedValue.textContent = data.speed + '/255';
    }

    // Update State
    if (data.state !== undefined) {
        const stateValue = document.getElementById('stateValue');
        const stateDot = document.getElementById('stateDot');
        stateValue.textContent = data.state;
        stateDot.style.background = getStateColor(data.state);
    }

    // Update Line Sensors
    if (data.line_sensors !== undefined) {
        updateLineSensors(data.line_sensors);
    }

    // Update Line Position
    if (data.line_position !== undefined) {
        document.getElementById('linePosition').textContent =
            (data.line_position > 0 ? '+' : '') + data.line_position;
    }

    // Update Ultrasonic Distance
    if (data.distance !== undefined) {
        const distValue = document.getElementById('ultrasonicValue');
        const distBar = document.getElementById('distanceBar');
        distValue.textContent = data.distance.toFixed(1);

        const percentage = Math.min(100, data.distance);
        distBar.style.width = percentage + '%';
    }
}

// ===== TARGET DATA UPDATE (FIXED) =====
function updateTargetData(data) {
    // ✅ ONLY update if in follow mode
    if (currentMode !== 'follow') {
        return;
    }

    // Show target-related UI elements
    const targetDistanceItem = document.getElementById('targetDistanceItem');
    const trackingStatusItem = document.getElementById('trackingStatusItem');
    const confidenceItem = document.getElementById('confidenceItem');

    targetDistanceItem.style.display = 'flex';
    trackingStatusItem.style.display = 'flex';
    confidenceItem.style.display = 'flex';

    // Update tracking status
    const trackingStatusValue = document.getElementById('trackingStatusValue');
    const trackingBadge = trackingStatusValue.querySelector('.tracking-badge');
    const targetStatusBadge = document.getElementById('targetStatusBadge');

    if (data.tracking) {
        trackingBadge.textContent = 'Tracking';
        trackingBadge.className = 'tracking-badge tracking-active';
        targetStatusBadge.textContent = 'Tracking';
        targetStatusBadge.className = 'tracking-badge tracking-active';
    } else {
        trackingBadge.textContent = 'Lost Target';
        trackingBadge.className = 'tracking-badge tracking-lost';
        targetStatusBadge.textContent = 'Lost';
        targetStatusBadge.className = 'tracking-badge tracking-lost';
    }

    // Update target distance
    if (data.target_distance !== undefined) {
        const targetDistanceValue = document.getElementById('targetDistanceValue');
        const targetDistanceInfo = document.getElementById('targetDistanceInfo');
        targetDistanceValue.textContent = data.target_distance.toFixed(1) + ' cm';
        targetDistanceInfo.textContent = data.target_distance.toFixed(1) + ' cm';
    }

    // Update confidence
    if (data.confidence !== undefined) {
        const confidenceBar = document.getElementById('confidenceBar');
        const confidenceValue = document.getElementById('confidenceValue');
        const targetConfidenceInfo = document.getElementById('targetConfidenceInfo');

        confidenceBar.style.width = data.confidence + '%';
        confidenceValue.textContent = data.confidence + '%';
        targetConfidenceInfo.textContent = data.confidence + '%';
    }

    // Update target position
    if (data.target_x !== undefined && data.target_y !== undefined) {
        document.getElementById('targetX').textContent = data.target_x;
        document.getElementById('targetY').textContent = data.target_y;

        // Update target box overlay
        updateTargetOverlay(data.target_x, data.target_y, data.target_w, data.target_h, data.tracking);
    }

    // Update target color info
    document.getElementById('targetColorInfo').textContent =
        selectedColor.charAt(0).toUpperCase() + selectedColor.slice(1);
}

// ===== NEW: UPDATE FOLLOW STATUS VISIBILITY =====
function updateFollowStatusVisibility() {
    const targetDistanceItem = document.getElementById('targetDistanceItem');
    const trackingStatusItem = document.getElementById('trackingStatusItem');
    const confidenceItem = document.getElementById('confidenceItem');
    const targetOverlay = document.getElementById('targetOverlay');

    if (currentMode === 'follow') {
        // Show follow-related status
        targetDistanceItem.style.display = 'flex';
        trackingStatusItem.style.display = 'flex';
        confidenceItem.style.display = 'flex';
    } else {
        // ✅ Hide follow-related status when NOT in follow mode
        targetDistanceItem.style.display = 'none';
        trackingStatusItem.style.display = 'none';
        confidenceItem.style.display = 'none';
        targetOverlay.style.display = 'none'; // Hide target box overlay
    }
}

// ===== UPDATE TARGET OVERLAY =====
function updateTargetOverlay(x, y, w, h, tracking) {
    const overlay = document.getElementById('targetOverlay');
    const targetBox = document.getElementById('targetBox');

    if (tracking && x !== undefined && y !== undefined) {
        overlay.style.display = 'block';

        // Convert normalized coordinates to pixels (assuming video frame size)
        // Adjust these values based on your actual video dimensions
        const videoWidth = 320;
        const videoHeight = 240;

        targetBox.style.left = (x * videoWidth) + 'px';
        targetBox.style.top = (y * videoHeight) + 'px';
        targetBox.style.width = (w * videoWidth) + 'px';
        targetBox.style.height = (h * videoHeight) + 'px';
    } else {
        overlay.style.display = 'none';
    }
}

// ===== STATE COLOR =====
function getStateColor(state) {
    const colors = {
        'IDLE': '#9e9e9e',
        'FOLLOWING LINE': '#4caf50',
        'FOLLOWING LANE': '#4caf50',
        'FOLLOWING TARGET': '#2196f3',
        'OBSTACLE DETECTED': '#ff9800',
        'STOPPED': '#f44336',
        'TURNING': '#2196f3',
        'MOVING FORWARD': '#4caf50',
        'MOVING BACKWARD': '#ff9800',
        'TURNING LEFT': '#2196f3',
        'TURNING RIGHT': '#2196f3',
        'AUTO MODE': '#9c27b0',
        'FOLLOW MODE': '#2196f3',
        'EMERGENCY STOP': '#f44336',
        'LANE LOST': '#f44336',
        'SEARCHING LANE': '#ff9800',
        'SCANNING LEFT': '#2196f3',
        'SCANNING RIGHT': '#2196f3',
        'RECOVERY FAILED': '#f44336'
    };
    return colors[state] || '#4caf50';
}

// ===== UPDATE LINE SENSORS =====
function updateLineSensors(sensors) {
    const sensorBars = document.querySelectorAll('.sensor-bar');
    sensors.forEach((value, index) => {
        if (index < sensorBars.length) {
            if (value) {
                sensorBars[index].classList.add('active');
            } else {
                sensorBars[index].classList.remove('active');
            }
        }
    });
}

// ===== ROBOT CONTROL COMMANDS =====
function sendCommand(command) {
    // Check mode before sending command
    if (currentMode !== 'manual') {
        console.log('Cannot send command in ' + currentMode + ' mode');
        addLogEntry(new Date().toLocaleTimeString(), 'WARNING',
            'Manual control disabled in ' + currentMode.toUpperCase() + ' mode');
        return;
    }

    fetch('/' + command)
        .then(response => {
            if (!response.ok) {
                throw new Error('Command failed');
            }
            return response.json();
        })
        .then(data => {
            console.log('Command sent:', command, 'Response:', data);
            if (data.status === 'error') {
                addLogEntry(new Date().toLocaleTimeString(), 'ERROR', data.message);
            }
        })
        .catch(error => {
            console.error('Error sending command:', error);
            addLogEntry(new Date().toLocaleTimeString(), 'ERROR',
                'Failed to send command: ' + command);
        });
}

// ===== EMERGENCY STOP =====
function emergencyStop() {
    if (confirm('Execute EMERGENCY STOP?')) {
        fetch('/emergency_stop')
            .then(response => response.json())
            .then(data => {
                console.log('Emergency stop executed:', data);
                addLogEntry(new Date().toLocaleTimeString(), 'WARNING',
                    'EMERGENCY STOP executed');
            })
            .catch(error => {
                console.error('Error executing emergency stop:', error);
            });
    }
}

// ===== MODE TOGGLE (FIXED) =====
function toggleMode() {
    const modeManual = document.getElementById('modeManual');
    const modeAuto = document.getElementById('modeAuto');
    const modeFollow = document.getElementById('modeFollow');

    if (modeManual.checked) {
        currentMode = 'manual';
    } else if (modeAuto.checked) {
        currentMode = 'auto';
    } else if (modeFollow.checked) {
        currentMode = 'follow';
    }

    // Send request to server
    fetch('/set_mode?mode=' + currentMode)
        .then(response => response.json())
        .then(data => {
            console.log('Mode changed to:', currentMode, data);
            addLogEntry(new Date().toLocaleTimeString(), 'INFO',
                'Mode changed to: ' + currentMode.toUpperCase());
            updateControlsState();
            updateFollowStatusVisibility(); // ✅ Update visibility immediately
        })
        .catch(error => {
            console.error('Error changing mode:', error);
            addLogEntry(new Date().toLocaleTimeString(), 'ERROR',
                'Failed to change mode');
        });
}

// ===== UPDATE MODE RADIO BUTTONS =====
function updateModeRadioButtons() {
    const modeManual = document.getElementById('modeManual');
    const modeAuto = document.getElementById('modeAuto');
    const modeFollow = document.getElementById('modeFollow');

    modeManual.checked = (currentMode === 'manual');
    modeAuto.checked = (currentMode === 'auto');
    modeFollow.checked = (currentMode === 'follow');
}

// ===== UPDATE CONTROLS STATE =====
function updateControlsState() {
    const controlPad = document.getElementById('controlPad');
    const buttons = controlPad.querySelectorAll('.ctrl-btn');
    const followSettings = document.getElementById('followSettings');

    // Disable manual control buttons if not in manual mode
    buttons.forEach(btn => {
        btn.disabled = (currentMode !== 'manual');
    });

    if (currentMode !== 'manual') {
        controlPad.style.opacity = '0.4';
    } else {
        controlPad.style.opacity = '1';
    }

    // Show/hide follow settings
    if (currentMode === 'follow') {
        followSettings.style.display = 'flex';
    } else {
        followSettings.style.display = 'none';
    }
}

// ===== COLOR SELECTION =====
function selectColor(color) {
    selectedColor = color;

    // Update UI
    const colorButtons = document.querySelectorAll('.color-btn');
    colorButtons.forEach(btn => {
        if (btn.dataset.color === color) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });

    // Send to server
    fetch('/set_follow_color?color=' + color)
        .then(response => response.json())
        .then(data => {
            console.log('Target color set to:', color, data);
            addLogEntry(new Date().toLocaleTimeString(), 'INFO',
                'Target color changed to: ' + color.toUpperCase());

            // Update target color info
            document.getElementById('targetColorInfo').textContent =
                color.charAt(0).toUpperCase() + color.slice(1);

            // Update target label
            document.getElementById('targetLabel').textContent =
                'Target: ' + color.charAt(0).toUpperCase() + color.slice(1);
        })
        .catch(error => {
            console.error('Error setting color:', error);
        });
}

// ===== SPEED SLIDER =====
const speedSlider = document.getElementById('speedSlider');
const speedDisplay = document.getElementById('speedDisplay');

speedSlider.addEventListener('input', function () {
    speedDisplay.textContent = this.value;
});

speedSlider.addEventListener('change', function () {
    const speed = this.value;
    fetch('/set_speed?value=' + speed)
        .then(response => response.json())
        .then(data => {
            console.log('Speed set to:', speed, data);
            addLogEntry(new Date().toLocaleTimeString(), 'INFO',
                'Speed adjusted to: ' + speed);
        })
        .catch(error => {
            console.error('Error setting speed:', error);
        });
});

// ===== FOLLOW DISTANCE SLIDER =====
const followDistanceSlider = document.getElementById('followDistanceSlider');
const followDistanceDisplay = document.getElementById('followDistanceDisplay');

followDistanceSlider.addEventListener('input', function () {
    followDistanceDisplay.textContent = this.value + ' cm';
});

followDistanceSlider.addEventListener('change', function () {
    followDistance = parseInt(this.value);
    fetch('/set_follow_distance?distance=' + followDistance)
        .then(response => response.json())
        .then(data => {
            console.log('Follow distance set to:', followDistance, data);
            addLogEntry(new Date().toLocaleTimeString(), 'INFO',
                'Follow distance set to: ' + followDistance + ' cm');
        })
        .catch(error => {
            console.error('Error setting follow distance:', error);
        });
});

// ===== KEYBOARD CONTROL =====
document.addEventListener('keydown', function (event) {
    // Only allow keyboard control in manual mode
    if (currentMode !== 'manual') {
        return;
    }

    switch (event.key) {
        case 'ArrowUp':
        case 'w':
        case 'W':
            sendCommand('forward');
            event.preventDefault();
            break;
        case 'ArrowDown':
        case 's':
        case 'S':
            sendCommand('backward');
            event.preventDefault();
            break;
        case 'ArrowLeft':
        case 'a':
        case 'A':
            sendCommand('left');
            event.preventDefault();
            break;
        case 'ArrowRight':
        case 'd':
        case 'D':
            sendCommand('right');
            event.preventDefault();
            break;
        case ' ':
            sendCommand('stop');
            event.preventDefault();
            break;
    }
});

// ===== KEY RELEASE HANDLER =====
document.addEventListener('keyup', function (event) {
    if (currentMode !== 'manual') {
        return;
    }

    const moveKeys = [
        'ArrowUp', 'w', 'W',
        'ArrowDown', 's', 'S',
        'ArrowLeft', 'a', 'A',
        'ArrowRight', 'd', 'D'
    ];

    if (moveKeys.includes(event.key)) {
        console.log("Key released: " + event.key + " → Sending STOP");
        sendCommand('stop');
        event.preventDefault();
    }
});

// ===== LOG MANAGEMENT =====
function addLogEntry(time, level, message) {
    const logContent = document.getElementById('logContent');

    const entry = document.createElement('div');
    entry.className = 'log-entry log-' + level.toLowerCase();
    entry.innerHTML = `
        <span class="log-time">${time}</span>
        <span class="log-level">${level}</span>
        <span class="log-message">${message}</span>
    `;

    // Add to top of log
    logContent.insertBefore(entry, logContent.firstChild);

    // Keep max 50 entries
    while (logContent.children.length > 50) {
        logContent.removeChild(logContent.lastChild);
    }
}

function clearLog() {
    if (confirm('Clear all log entries?')) {
        fetch('/clear_log')
            .then(response => response.json())
            .then(data => {
                console.log('Log cleared:', data);
                document.getElementById('logContent').innerHTML = '';
                addLogEntry(new Date().toLocaleTimeString(), 'INFO',
                    'Log cleared by user');
            })
            .catch(error => {
                console.error('Error clearing log:', error);
            });
    }
}

// ===== UPTIME COUNTER =====
function updateUptime() {
    const elapsed = Date.now() - startTime;
    const hours = Math.floor(elapsed / 3600000);
    const minutes = Math.floor((elapsed % 3600000) / 60000);

    document.getElementById('uptimeValue').textContent =
        hours + 'h ' + minutes + 'm';
}

// Update uptime every minute
setInterval(updateUptime, 60000);

// ===== VIDEO STREAM ERROR HANDLING =====
document.getElementById('videoStream').addEventListener('error', function () {
    console.error('Error loading video stream');
    addLogEntry(new Date().toLocaleTimeString(), 'ERROR',
        'Video stream connection lost');
});

// ===== VISUAL ODOMETRY UI UPDATES =====
setInterval(updateOdometryStatus, 1000);

async function updateOdometryStatus() {
    try {
        const response = await fetch('/api/odometry/status');

        if (!response.ok) {
            document.getElementById('vo-distance').textContent = 'N/A';
            document.getElementById('vo-velocity').textContent = 'N/A';
            document.getElementById('vo-quality').textContent = 'N/A';
            return;
        }

        const data = await response.json();

        if (data.enabled) {
            document.getElementById('vo-distance').textContent =
                data.distance_cm.toFixed(1) + ' cm';

            const velocity = Math.sqrt(
                Math.pow(data.velocity.x_cm_s, 2) +
                Math.pow(data.velocity.y_cm_s, 2)
            );
            document.getElementById('vo-velocity').textContent =
                velocity.toFixed(1) + ' cm/s';

            const qualityBadge = document.getElementById('vo-quality');
            const quality = data.quality;
            qualityBadge.textContent = quality.toFixed(2);

            if (quality > 0.7) {
                qualityBadge.className = 'badge badge-success';
            } else if (quality > 0.4) {
                qualityBadge.className = 'badge badge-warning';
            } else {
                qualityBadge.className = 'badge badge-danger';
            }

            document.getElementById('vo-pos-x').textContent =
                data.position.x_cm.toFixed(1);
            document.getElementById('vo-pos-y').textContent =
                data.position.y_cm.toFixed(1);
        }
    } catch (error) {
        console.error('Error updating odometry:', error);
    }
}

async function resetOdometry() {
    if (!confirm('Reset odometry position to (0, 0)?')) {
        return;
    }

    try {
        const response = await fetch('/api/odometry/reset', {
            method: 'POST'
        });

        const data = await response.json();

        if (data.status === 'success') {
            alert('✅ Odometry reset successfully!');
            updateOdometryStatus();
        } else {
            alert('❌ Failed to reset: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error resetting odometry:', error);
        alert('❌ Network error');
    }
}

function showCalibrationDialog() {
    const distance = prompt('Enter distance traveled (cm):', '20');
    const pixels = prompt('Enter measured pixels:', '400');

    if (distance && pixels) {
        calibrateOdometry(parseFloat(distance), parseFloat(pixels));
    }
}

async function calibrateOdometry(distance_cm, measured_pixels) {
    try {
        const response = await fetch('/api/odometry/calibrate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                distance_cm: distance_cm,
                measured_pixels: measured_pixels
            })
        });

        const data = await response.json();

        if (data.status === 'success') {
            alert(`✅ Calibration successful!\n${data.message}`);
        } else {
            alert('❌ Calibration failed: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error calibrating:', error);
        alert('❌ Network error');
    }
}

// ===== INITIALIZATION =====
window.addEventListener('load', function () {
    console.log('Dashboard initialized');
    updateUptime();
    updateControlsState();
    updateFollowStatusVisibility(); // ✅ Set initial visibility

    // Initial log entry
    addLogEntry(new Date().toLocaleTimeString(), 'INFO',
        'LogisticsBot Control Panel initialized');
});