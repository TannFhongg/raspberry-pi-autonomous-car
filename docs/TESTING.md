# Testing

## Testing Philosophy

This repository mixes:

- runtime-facing smoke tests
- hardware bring-up scripts
- calibration tools
- one Python unit/integration test file

That is common in robotics projects, but it also means not every script has the same level of maturity. This guide separates the active path from legacy utilities so you can present the project honestly.

## Recommended Test Order

1. Camera connectivity
2. Arduino serial connectivity
3. Lane detection on a still frame
4. Lane tuning dashboard
5. PID sanity check
6. Motor test with wheels lifted
7. Full system integration test

## Active Test and Calibration Scripts

| File | What it does | Prerequisites | Notes |
| --- | --- | --- | --- |
| [`review_tool/test_camera.py`](../review_tool/test_camera.py) | Captures frames, measures rough FPS, saves a sample image | Pi camera + Picamera2 | Good first hardware smoke test |
| [`review_tool/test_arduino.py`](../review_tool/test_arduino.py) | Scans serial ports and validates JSON `PING` communication | Arduino over USB | Best match to current serial protocol |
| [`review_tool/test_lane_detection.py`](../review_tool/test_lane_detection.py) | Runs lane detection on a static image or lane-width calibration | Test image | Writes `review_tool/test_lane_result.jpg` |
| [`calibrate_fixed.py`](../calibrate_fixed.py) | Measures `lane_width_pixels` from one image using current lane settings | Test image | Useful when you want a more explicit calibration artifact |
| [`dashboard_server.py`](../dashboard_server.py) | Live lane tuning dashboard | Camera | Best tool for ROI/Canny/Hough tuning |
| [`review_tool/test_pid.py`](../review_tool/test_pid.py) | Simulates PID response and saves plots | `matplotlib` | Good for controller explanation in a portfolio |
| [`review_tool/test_motor.py`](../review_tool/test_motor.py) | Exercises forward/backward/turn/individual wheel commands via ArduinoDriver | Arduino + motors | Run with wheels lifted first |
| [`review_tool/test_model.py`](../review_tool/test_model.py) | Benchmarks model inference on a still image | YOLO model assets | Good for validating model load and inference time |
| [`test_follow.py`](../test_follow.py) | Follow-mode target-size tuning dashboard | Camera + model assets | Default port conflicts with `dashboard_server.py` |
| [`test_mode_auto.py`](../test_mode_auto.py) | Sign-size threshold tuning dashboard | Camera + model assets | Writes thresholds back to YAML |
| [`review_tool/test_full_system.py`](../review_tool/test_full_system.py) | End-to-end camera -> lane -> PID -> Arduino integration test | Full hardware stack | Use only in a safe physical setup |

## Quick Smoke Tests

These smaller utilities still exist and can be useful:

| File | Purpose | Notes |
| --- | --- | --- |
| [`tools/test_camera.py`](../tools/test_camera.py) | Minimal camera start/capture smoke test | Very lightweight |
| [`tools/test_arduino.py`](../tools/test_arduino.py) | Generic serial connectivity diagnostic | Uses a simpler text-oriented flow than the `review_tool` version |
| [`tools/generate_road.py`](../tools/generate_road.py) | Generates synthetic road images | Useful for quick algorithm experiments |

## Unit and Logic Tests

The repository contains one main Python test module:

| File | Coverage area | Current status |
| --- | --- | --- |
| [`tests/test_robot_logic.py`](../tests/test_robot_logic.py) | Camera buffering, YUV420 handling, lane width logic, route behavior, dashboard validation | Partially stale because it imports missing `tune_lane_web` |

Useful test cases inside that file include:

- YUV420 detection uses the actual frame width
- changing `lane_width_pixels` changes single-lane estimation
- blur-kernel validation normalizes even values
- `/set_follow_distance` references are intentionally absent

This makes the file valuable as a specification reference even though it needs cleanup before it becomes a reliable passing suite again.

## Model and Performance Checks

### Lane detection test

```bash
python review_tool/test_lane_detection.py --image review_tool/test_camera_frame.jpg
```

### Lane calibration test

```bash
python review_tool/test_lane_detection.py --image review_tool/test_camera_frame.jpg --calibrate
```

### PID response / timing-style analysis

```bash
python review_tool/test_pid.py
```

This is not a real-time hardware timing probe, but it does provide a useful controller response simulation and plot output.

### YUV / camera format coverage

There is no standalone `test_yuv.py`, but YUV-related behavior is covered by:

- `review_tool/test_camera.py`
- `tools/test_camera.py`
- `tests/test_robot_logic.py` cases that verify YUV420 conversion and width handling

### Model benchmark

```bash
python review_tool/test_model.py --image <path-to-test-image>
```

## Scripts With Known Legacy Gaps

These files exist, but they should be documented as legacy or incomplete:

| File | Gap |
| --- | --- |
| [`tools/test_motor.py`](../tools/test_motor.py) | depends on missing `drivers/motor/l298n_driver.py` |
| [`tools/calibrate_vo.py`](../tools/calibrate_vo.py) | depends on missing `perception.visual_odometry` |
| [`tools/test_yolo_ncnn.py`](../tools/test_yolo_ncnn.py) | expects `data/models/best_ncnn_model`, which differs from the main runtime path |

That does not make the repo weak. It simply means the project has some cleanup debt, which is normal for an evolving robotics codebase.

## Suggested Demo Validation Workflow

Before recording a portfolio video:

1. Run `review_tool/test_camera.py`
2. Run `review_tool/test_arduino.py`
3. Recheck lane calibration on a fresh image
4. Tune lane parameters with `dashboard_server.py` if the lighting changed
5. Run a low-speed autonomous pass
6. Record the main dashboard plus a real track run

## Output Files Worth Saving

Helpful artifacts created by the current tooling:

- `review_tool/test_camera_frame.jpg`
- `review_tool/test_lane_result.jpg`
- `review_tool/test_model_result.jpg`
- `review_tool/test_pid_result.png`
- `calibration_result_fixed.jpg`

Because the repo currently ignores `*.jpg` and `*.png` globally outside `docs/assets/`, treat those as local debug artifacts unless you intentionally move selected screenshots into `docs/assets/` for portfolio presentation.
