# Setup

## Scope

This guide is based on the files that exist in this workspace today. It focuses on the supported path:

- Raspberry Pi runtime
- Arduino Uno over USB serial
- Picamera2 camera input
- optional YOLO and IMU dependencies installed separately where needed

## 1. Prepare the Raspberry Pi

Install a current Raspberry Pi OS image and make sure you have:

- Python 3
- `pip`
- `venv`
- Git
- camera access enabled

Recommended package baseline:

```bash
sudo apt update
sudo apt install -y git python3-pip python3-venv
```

If you plan to use the optional IMU path, also enable I2C in `raspi-config`.

## 2. Clone the Project

```bash
git clone <your-repo-url>
cd Viet
```

If your repository name changes later, keep the commands generic. The source still uses `LogisticsBot` internally in some comments, but the setup does not depend on that name.

## 3. Create a Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 4. Install Python Dependencies

Install the base requirements first:

```bash
pip install -r requirements.txt
```

### Optional dependencies for features not listed in `requirements.txt`

The current source imports extra packages that are not pinned in the repo requirements file:

- `ultralytics` for YOLO-based follow/sign features
- `smbus2` for MPU-6050 support
- `matplotlib` for `review_tool/test_pid.py`

Install them only if you need those features:

```bash
pip install ultralytics smbus2 matplotlib
```

### Picamera2 note

If `picamera2` does not install cleanly from `pip` on your Pi image, use the Raspberry Pi OS package path instead and then reinstall the remaining Python packages inside the virtual environment.

## 5. Enable and Check the Camera

Make sure the CSI camera is enabled and detected:

```bash
libcamera-hello --list-cameras
```

Then run the repo camera smoke test:

```bash
python review_tool/test_camera.py
```

Expected outcome:

- camera starts
- frames can be captured
- `review_tool/test_camera_frame.jpg` is written

## 6. Upload the Arduino Firmware

Open [`arduino_firmware/arduino_firmware.ino`](../arduino_firmware/arduino_firmware.ino) in Arduino IDE and:

1. Select `Arduino Uno`
2. Install `ArduinoJson`
3. Upload the sketch over USB

The firmware is configured for:

- `115200` baud
- JSON commands over serial
- motor watchdog timeout of `2 seconds`

## 7. Configure the Serial Port

The default repo config is:

```yaml
arduino:
  port: /dev/ttyACM0
  baudrate: 115200
```

Check the actual device path on the Pi:

```bash
ls /dev/ttyACM* /dev/ttyUSB*
```

If needed, update [`config/hardware_config.yaml`](../config/hardware_config.yaml).

### Serial permissions

If you get permission errors:

```bash
sudo usermod -a -G dialout $USER
```

Then log out and back in.

## 8. Validate the Arduino Link

Run one of the serial diagnostics:

```bash
python review_tool/test_arduino.py
```

or

```bash
python tools/test_arduino.py
```

The `review_tool` version matches the current JSON `PING` protocol more closely.

## 9. Review Runtime Configuration

Before the first full run, inspect these sections in `config/hardware_config.yaml`:

- `control_mode`
- `arduino.port`
- `sensors.camera`
- `ai.lane_detection`
- `lane_following.pid`
- `follow_mode`

Recommended first checks:

- confirm the correct serial port
- confirm camera resolution and format
- confirm `lane_width_pixels` matches your track calibration
- reduce `base_speed` for first motion tests if needed

## 10. Run the Main Application

```bash
python main.py
```

The primary dashboard runs on:

- `http://<raspberry-pi-ip>:5000`

## 11. Optional Tuning Tools

### Lane tuning dashboard

```bash
python dashboard_server.py
```

Default:

- port `5001`

### Follow target tuning dashboard

```bash
python test_follow.py --port 5003
```

Why `5003` here:

- `test_follow.py` defaults to `5001`
- `dashboard_server.py` also defaults to `5001`
- changing one port avoids a collision

### Auto sign-size tuner

```bash
python test_mode_auto.py
```

Default:

- port `5002`

## 12. Common Setup Issues

### Camera opens in one tool but not another

Cause:

- another process is already holding the camera

Fix:

- stop other camera apps
- reboot the Pi if the camera is left in a bad state

### Main app starts but follow/sign features do nothing

Cause:

- model assets are missing from `models/best_ncnn_model`

Fix:

- copy the model assets into the expected path
- verify that `ObjectDetector` logs a successful model load

### `review_tool/test_pid.py` fails

Cause:

- `matplotlib` is not installed

Fix:

```bash
pip install matplotlib
```

### IMU support does not initialize

Cause:

- `smbus2` missing, I2C disabled, or MPU-6050 wiring issue

Fix:

- enable I2C
- install `smbus2`
- run `sudo i2cdetect -y 1`

### Unit tests fail immediately on import

Cause:

- `tests/test_robot_logic.py` references `tune_lane_web`, which is not present in this workspace

Fix:

- treat the test suite as partially stale until that dependency is restored or the test file is updated
