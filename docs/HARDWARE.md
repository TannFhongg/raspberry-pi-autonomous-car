# Hardware

## Current Hardware Assumption

The active codebase is built around:

- Raspberry Pi as the main compute node
- Arduino Uno as a serial-connected motor controller
- L298N motor driver
- two DC motors
- Raspberry Pi Camera Module

The current Arduino firmware is explicitly marked as a camera-only build. It does not use IR line sensors or ultrasonic ranging in the shipped control path.

## Required Hardware

| Component | Role | Notes |
| --- | --- | --- |
| Raspberry Pi 5 | Main compute board | Hosts Flask, camera, lane detection, and high-level control |
| Arduino Uno | Motor control coprocessor | Receives JSON commands over serial |
| L298N driver | Dual H-bridge motor driver | Drives left/right DC motors |
| 2x DC geared motors | Locomotion | Wheel polarity may need to be swapped during bring-up |
| Raspberry Pi Camera Module | Vision input | Required for lane following and model-assisted perception |
| Chassis + wheels | Mechanical platform | Mount camera rigidly to reduce perception drift |
| Motor battery supply | Motor power rail | Keep enough headroom for startup current |
| 5 V supply for Pi | Logic/compute power | Use a stable supply sized for Pi + camera load |
| USB cable to Arduino | Serial link | Current config expects `/dev/ttyACM0` style USB serial |

## Optional Hardware

| Component | Value | Status |
| --- | --- | --- |
| MPU-6050 IMU | Enables `smart_turn(...)` with yaw feedback | Optional, supported in code |
| External power distribution / buck converter | Cleaner power architecture | Recommended for stable field testing |
| Encoder-equipped motors | Better closed-loop odometry | Not implemented in current code |
| Physical emergency power cut | Safer test workflow | Strongly recommended |

## Wiring Overview

### Arduino Uno to L298N

These pin mappings come directly from [`arduino_firmware/arduino_firmware.ino`](../arduino_firmware/arduino_firmware.ino):

| Arduino pin | L298N pin | Function |
| --- | --- | --- |
| `D9` | `ENA` | Left motor PWM |
| `D2` | `IN1` | Left motor direction A |
| `D3` | `IN2` | Left motor direction B |
| `D10` | `ENB` | Right motor PWM |
| `D4` | `IN3` | Right motor direction A |
| `D5` | `IN4` | Right motor direction B |
| `GND` | `GND` | Common ground |

### L298N to motors and power

| L298N pin | Connection |
| --- | --- |
| `OUT1`, `OUT2` | Left motor |
| `OUT3`, `OUT4` | Right motor |
| `12V` / motor VIN | Motor battery positive |
| `GND` | Motor battery negative |

Implementation note:

- If your L298N board ships with enable jumpers installed, remove them if you want Arduino PWM on `ENA` and `ENB` to control speed.

## Raspberry Pi to Arduino Communication

## Recommended path

Use a USB connection between the Pi and the Arduino.

Why this is the recommended path:

- `hardware_config.yaml` defaults to `/dev/ttyACM0`
- `ArduinoDriver` opens a serial device path, not GPIO UART directly
- USB serial is the clearest path to reproduce from the current repo

### Current software expectation

| Item | Expected value |
| --- | --- |
| Port | `/dev/ttyACM0` by default |
| Baud rate | `115200` |
| Protocol | newline-delimited JSON |

### Direct UART note

Direct GPIO UART between the Pi and the Arduino is possible in principle, but it is not the documented or verified integration path in this repo. If you choose that route, handle logic-level safety carefully and update the Linux serial device configuration accordingly.

## Camera Setup

| Item | Recommendation |
| --- | --- |
| Mounting position | Centered on chassis if possible |
| Mounting angle | Slight downward angle so the lower half of the frame captures the track |
| Mount rigidity | High priority; vibration directly hurts lane detection quality |
| Cable routing | Secure the ribbon cable to avoid intermittent camera faults |

The default config currently uses:

- processing resolution: `960 x 720`
- sensor output size: `1640 x 1232`
- framerate target: `30 FPS`
- format: `RGB888`

## Optional IMU Wiring

The IMU code in [`perception/imu_sensor_fusion.py`](../perception/imu_sensor_fusion.py) logs the following expected MPU-6050 wiring:

| MPU-6050 pin | Raspberry Pi pin |
| --- | --- |
| `VCC` | `3.3V` |
| `GND` | `GND` |
| `SDA` | `Pin 3` |
| `SCL` | `Pin 5` |

I2C address used by the code:

- `0x68` on bus `1`

## Power Considerations

Recommended power strategy:

- power the Raspberry Pi from a dedicated, stable 5 V source
- power motors from a separate battery rail sized for stall current
- tie grounds together
- keep camera and logic wiring away from motor noise when possible

Why this matters:

- motor brownouts can crash the Pi
- noisy power can corrupt serial communication
- unstable voltage will make camera bring-up harder to diagnose

## Safety Notes

- Test with wheels lifted before the first ground run.
- Keep an accessible physical stop option nearby.
- Expect wheel polarity to need adjustment after first assembly.
- Do not assume the emergency stop in software is a substitute for safe bench testing.
- The Arduino watchdog stops motors if the Pi stops sending commands, but that should be treated as a safety layer, not the only protection.

## Known Hardware Ambiguities

These items appear in parts of the repo history or older local notes, but are not part of the current validated path:

- IR line sensor arrays
- HC-SR04 ultrasonic sensing
- direct GPIO motor control via `l298n_driver.py`
- bundled YOLO model files in the repository

## Wiring Diagram Placeholder

Add your own photo or diagram here for portfolio presentation:

- `docs/assets/hardware_setup.jpg`

Suggested content:

- top-down chassis photo
- labeled Pi, Arduino, L298N, camera, and battery
- a small callout showing the USB serial path between Pi and Arduino
