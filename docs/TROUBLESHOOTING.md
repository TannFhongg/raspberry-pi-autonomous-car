# Troubleshooting

## Camera Not Detected

Symptoms:

- `main.py` fails to enter auto or follow mode
- `review_tool/test_camera.py` cannot start the camera
- `/video_feed` or `/debug_feed` returns a blank/error stream

Checks:

- confirm the ribbon cable is fully seated
- run `libcamera-hello --list-cameras`
- make sure no other process is already using the camera
- reboot after changing camera settings

Likely fixes:

- reseat the camera cable
- close other camera apps
- reinstall or verify `picamera2`
- reduce concurrent camera-consuming tools

## Arduino Serial Not Connected

Symptoms:

- `main.py` exits during hardware initialization
- `review_tool/test_arduino.py` cannot find a working port
- speed/motion commands have no physical effect

Checks:

- `ls /dev/ttyACM* /dev/ttyUSB*`
- confirm `arduino.port` in `config/hardware_config.yaml`
- confirm baud rate `115200`
- verify the Arduino firmware is uploaded

Likely fixes:

- change the port in config to the actual device
- reconnect USB
- add your user to the `dialout` group
- use `review_tool/test_arduino.py` to validate the JSON `PING` path

## Motors Not Moving

Symptoms:

- mode changes succeed but the robot does not move
- `review_tool/test_motor.py` connects, but wheels stay still

Checks:

- L298N power LED is on
- motor battery is connected
- ENA and ENB PWM lines are wired correctly
- motor wires are connected to `OUT1`-`OUT4`
- Arduino firmware is the expected camera-only motor controller build

Likely fixes:

- verify the pin mapping from `arduino_firmware.ino`
- test with wheels lifted
- check for undervoltage on the motor supply
- confirm the controller is not in emergency stop

## Wrong Motor Direction

Symptoms:

- forward commands spin the robot backward
- left/right steering is mirrored

Checks:

- run `review_tool/test_motor.py`
- observe left and right wheel behavior independently

Likely fixes:

- swap motor leads on the L298N outputs
- verify left/right wiring matches the firmware assumptions

## Lane Detection Is Unstable

Symptoms:

- frequent `NO_LANE`
- oscillating error values
- line overlay jumps from frame to frame

Checks:

- camera is rigidly mounted
- track lighting is consistent
- ROI actually covers both lane boundaries
- `lane_width_pixels` matches your real track

Likely fixes:

- recalibrate with `review_tool/test_lane_detection.py --calibrate`
- use `dashboard_server.py` to tune ROI/Canny/Hough values
- lower speed while tuning
- adjust `camera_offset` after mechanical alignment is as good as possible

## Dashboard Not Loading

Symptoms:

- browser cannot connect
- white page or stream never updates

Checks:

- confirm which process is running on which port
- `main.py` uses `5000`
- `dashboard_server.py` uses `5001`
- `test_follow.py` also defaults to `5001`
- `test_mode_auto.py` uses `5002`

Likely fixes:

- avoid port collisions
- check the Pi IP address with `hostname -I`
- test locally with `curl http://localhost:5000`

## High Latency or Low FPS

Symptoms:

- delayed video
- sluggish control response
- large lag before sign/follow updates

Checks:

- confirm camera format and resolution in `hardware_config.yaml`
- check CPU load on the Pi
- verify whether YOLO features are enabled

Likely fixes:

- reduce concurrent dashboards
- lower the processing resolution if needed
- use the classical lane-following path first, then add model features
- confirm the Pi is not thermally throttling

## Permission Issues on Raspberry Pi

Symptoms:

- serial permission denied
- I2C access failures for IMU

Likely fixes:

```bash
sudo usermod -a -G dialout $USER
sudo raspi-config
```

Then:

- re-login after changing groups
- enable I2C if using MPU-6050

## PID Tuning Issues

### Oscillation

Usually means:

- `Kp` too high
- `Kd` too low
- speed too high for the current tune

Try:

- lower `Kp`
- raise `Kd`
- lower `base_speed`

### Slow response

Usually means:

- `Kp` too low
- speed too low
- lane signal too noisy or delayed

Try:

- raise `Kp` gradually
- verify lane detection quality before changing PID too much

### Overshooting in curves

Usually means:

- aggressive steering correction
- delayed perception from poor frame quality or too much lag

Try:

- reduce `Kp`
- increase `Kd`
- lower speed and retest

## Follow Mode Does Not Track Anything

Symptoms:

- UI stays in search mode
- target confidence never rises

Checks:

- model files exist at `models/best_ncnn_model`
- the selected target color matches the trained class names
- lighting and target appearance match the training data assumptions

Likely fixes:

- restore the missing model assets
- confirm classes such as `red_color`, `green_color`, `blue_color`, `yellow_color`
- lower the tuner confidence threshold temporarily for debugging

## Traffic Sign Features Do Not Trigger

Symptoms:

- signs appear in view but auto mode does not react

Checks:

- model assets are present
- sign classes match the code paths in `AutoModeController`
- `dist_prepare` and `dist_execute` are realistic for your camera angle and track

Likely fixes:

- use `test_mode_auto.py` to measure sign size thresholds
- confirm the relevant classes exist in the trained model

## Unit Tests Fail Before Running

Symptoms:

- `tests/test_robot_logic.py` errors on import

Cause:

- it imports a missing module named `tune_lane_web`

Recommended handling:

- treat the unit test suite as partially stale
- use the `review_tool/` scripts and targeted runtime checks until the missing test dependency is restored

## Legacy Tool Mismatches

Known legacy/stale references:

- `tools/test_motor.py` expects `drivers/motor/l298n_driver.py`, which is not present
- `tools/calibrate_vo.py` expects `perception.visual_odometry`, which is not present
- `tools/test_yolo_ncnn.py` uses `data/models/...`, while the main runtime uses `models/...`

These are good cleanup candidates, but they should not be presented as fully verified parts of the current runtime.
