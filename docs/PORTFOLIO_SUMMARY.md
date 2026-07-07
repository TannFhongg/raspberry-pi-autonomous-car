# Portfolio Summary

## Two-Line Project Summary

Autonomous mini car built on a Raspberry Pi + Arduino split architecture, combining camera-based lane following, PID control, and a Flask dashboard for runtime monitoring and tuning.  
The project shows practical embedded robotics engineering across perception, serial communication, motor control, calibration, and safety-aware system integration.

## Technical Highlights

- Raspberry Pi hosts perception, web UI, and high-level control logic.
- Arduino Uno acts as a dedicated motor-control coprocessor over serial JSON.
- Lane following uses a classical computer-vision pipeline with ROI, Canny, Hough, and configurable calibration.
- Follow mode uses YOLO detections filtered by color-class labels and dual PID loops for centering and distance control.
- Traffic sign reactions are implemented in the auto controller as model-assisted, threshold-based behaviors.
- Safety is reinforced with watchdog logic on both the Pi side and Arduino firmware side.

## Skills Demonstrated

- Python robotics application development
- Flask and Socket.IO dashboard integration
- Raspberry Pi camera and Linux device bring-up
- Arduino firmware development and serial protocols
- PID tuning and closed-loop motor control
- Computer vision debugging and calibration
- Embedded system safety thinking
- Documentation and technical communication

## Problems Solved

- Shared camera access across web streaming and control loops
- Safe serial-based separation between compute and actuation
- Lane detection tuning for real-world lighting and track geometry
- Runtime mode switching between lane following and vision-based follow mode
- Automatic motor shutdown when communication is lost

## Suggested CV Bullet Points

- Built an autonomous mini car using Raspberry Pi and Arduino, with a distributed control architecture separating perception from motor actuation.
- Implemented PID-based lane following using a camera pipeline with ROI filtering, Canny edge detection, and Hough line estimation.
- Developed a Flask + Socket.IO dashboard for live video streaming, runtime control, logging, and parameter tuning.
- Integrated serial JSON communication and watchdog safety mechanisms between Raspberry Pi and Arduino motor controller firmware.
- Created calibration and hardware bring-up tools for lane tuning, target tracking, and end-to-end system testing.

## Suggested GitHub Project Description

Raspberry Pi + Arduino autonomous mini car with camera-based lane following, YOLO-assisted perception, PID control, and Flask dashboard tooling.

## Suggested GitHub About Line

Camera-based autonomous mini car using Raspberry Pi + Arduino distributed control, PID lane following, and Flask monitoring tools.

## Suggested Repository Topics

`raspberry-pi`, `arduino`, `autonomous-car`, `computer-vision`, `pid-controller`, `flask-dashboard`, `robotics`, `embedded-systems`

## Suggested LinkedIn Post

I built a small autonomous robotics project that combines Raspberry Pi perception, Arduino motor control, PID-based lane following, and a Flask dashboard for monitoring and tuning. One of the most valuable parts of the project was working through real integration issues across camera pipelines, serial communication, safety watchdogs, and calibration on physical hardware.
