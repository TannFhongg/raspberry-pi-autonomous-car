# Calibration

## Goal

Calibration in this project is mostly about making classical vision and PID control behave consistently on your real robot:

- lane width must match your track
- camera position must match the assumptions in the ROI
- PID gains must match your chassis dynamics
- speed limits must match your available traction and power

## Before You Start

Use a safe test setup:

- lift the wheels for first controller checks
- use a low `base_speed`
- clear the floor around the robot
- keep a hand near the power switch or unplug point

## Lane Detection Calibration

## What you are tuning

The lane detector relies on:

- ROI boundaries
- Canny thresholds
- Hough thresholds
- blur kernel
- `lane_width_pixels`
- `camera_offset`

These live under `ai.lane_detection` in [`config/hardware_config.yaml`](../config/hardware_config.yaml).

## Recommended workflow

### 1. Capture a representative frame

```bash
python review_tool/test_camera.py
```

This writes:

- `review_tool/test_camera_frame.jpg`

### 2. Measure lane width from a still image

You have two useful options:

```bash
python review_tool/test_lane_detection.py --image review_tool/test_camera_frame.jpg --calibrate
```

or

```bash
python calibrate_fixed.py review_tool/test_camera_frame.jpg
```

Important note:

- the current calibration helpers assume a real lane width of `25 cm`
- if your physical track is different, update that assumption before trusting the scale factor

### 3. Tune the live lane parameters

```bash
python dashboard_server.py
```

Use the dashboard to adjust:

- ROI window
- Canny thresholds
- Hough sensitivity
- blur kernel
- camera offset

Save tuned values back into `hardware_config.yaml` from the dashboard once the overlay is stable.

## PID Tuning Explanation

The lane controller uses a standard PID loop:

- `Kp`: reacts to current error
- `Ki`: corrects long-term bias
- `Kd`: damps oscillation and sharpens transient behavior

Current PID config lives under:

- `lane_following.pid`

## Practical tuning order

1. Start with `Ki = 0`.
2. Increase `Kp` until the robot responds decisively.
3. Add `Kd` until oscillation reduces.
4. Add only a small `Ki` if the robot drifts consistently to one side even after mechanical and camera alignment are corrected.

## Camera Position Adjustment

Camera placement has a direct impact on the lane detector.

Check these items first:

- camera is centered left-to-right on the chassis as closely as possible
- camera angle is repeatable and rigid
- camera sees enough track in the lower half of the frame
- left and right lane boundaries both enter the ROI when the robot is centered

If the robot is physically centered but the measured error is consistently biased:

- first inspect camera mounting
- then use `camera_offset` as a software trim

## Speed Tuning

Speed interacts strongly with stability.

Tune in this order:

1. Lower `base_speed` until the robot can complete a stable straight run.
2. Confirm steering still has enough authority in curves.
3. Increase speed in small steps.
4. Revisit `Kp` and `Kd` after each meaningful speed increase.

Relevant config:

- `lane_following.base_speed`
- `lane_following.max_speed`
- `lane_following.min_speed`

## Safe Testing Procedure

Use this progression:

1. Bench test with wheels lifted.
2. Low-speed straight-line test.
3. Gentle curve test.
4. Repeated start-stop runs.
5. Full autonomous run only after the first four are stable.

Do not jump straight to the highest configured speed.

## Recommended End-to-End Calibration Workflow

1. Confirm camera and Arduino connectivity.
2. Capture a representative frame.
3. Calibrate `lane_width_pixels`.
4. Tune ROI and edge/line thresholds live.
5. Set a conservative `base_speed`.
6. Tune `Kp` and `Kd`.
7. Add `Ki` only if needed.
8. Validate on the real track under the same lighting you expect during demo runs.

## Symptom Guide

| Symptom | Likely cause | First things to change |
| --- | --- | --- |
| Oscillation / weaving | `Kp` too high, `Kd` too low, speed too high | reduce `Kp`, increase `Kd`, lower speed |
| Slow response | `Kp` too low, speed too low, ROI too conservative | increase `Kp`, review ROI |
| Overshoot in curves | aggressive correction or delayed perception | reduce `Kp`, increase `Kd`, lower speed |
| Loses lane intermittently | lighting, ROI, thresholds, camera vibration | tune ROI/Canny/Hough, improve mounting |
| Constant bias to one side | camera not centered or offset not corrected | re-center camera, tune `camera_offset` |
| Finds only one lane edge | `lane_width_pixels` wrong or one side outside ROI | re-calibrate lane width, widen ROI |

## Follow Mode Calibration

Follow mode is tuned separately from lane following.

Use:

```bash
python test_follow.py --port 5003
```

Tune:

- `target_size`
- `size_tolerance`
- minimum confidence

This tool is especially useful because the controller uses object size as a stand-in for distance.

## Traffic Sign Threshold Calibration

If you use sign handling in auto mode:

```bash
python test_mode_auto.py
```

Tune:

- `dist_prepare`
- `dist_execute`

These thresholds are based on bounding-box size, not physical distance sensors.

## Known Calibration Constraints

- model assets are not included in this workspace
- lighting changes will affect classical lane detection quality
- `dashboard_server.py` and `test_follow.py` both default to port `5001`
- some legacy tools in `tools/` reference missing modules and should not be treated as the main calibration path
