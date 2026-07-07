# Roadmap

## Roadmap Principles

The roadmap below is intentionally realistic. It is based on the current codebase and its visible gaps, not on generic robotics wishlist items presented as if they already exist.

## Short-Term Improvements

- Consolidate or retire legacy scripts that reference missing modules such as `tune_lane_web`, `visual_odometry`, and `l298n_driver.py`.
- Package or document the expected YOLO model assets so follow mode and sign logic are easier to reproduce.
- Resolve default port conflicts between `dashboard_server.py` and `test_follow.py`.
- Add a small startup self-check for camera, serial, and model availability before mode activation.
- Improve log messages around controller start failures and model-loading failures.
- Capture and commit real demo media under `docs/assets/`.

## Mid-Term Improvements

- Add repeatable calibration presets for different tracks or lighting conditions.
- Expose richer real-time telemetry in the main dashboard, such as PID terms, frame rate, and controller state.
- Add a reproducible unit-test path that does not rely on missing modules.
- Standardize model path handling across the main runtime and test scripts.
- Add lightweight CI for pure-Python checks, config validation, and documentation linting.
- Restore an intentional manual teleoperation mode if it is still a project goal.

## Long-Term Improvements

- Improve traffic sign detection quality with a better-labeled dataset and clearer evaluation workflow.
- Add obstacle-awareness or emergency stop cues from extra sensing if the hardware grows beyond the current camera-only architecture.
- Replace heuristic distance estimation in follow mode with a more explicit distance model or calibrated depth cue.
- Explore ROS 2 migration if the project grows into a multi-node robotics stack.
- Add localization or SLAM only if the hardware and project scope justify it.
- Evolve the dashboard from a demo control panel into a telemetry and experiment console.

## Skill-Building Milestones

If the project is being presented for internships or fresher roles, these milestones add strong portfolio value:

- documented calibration workflow
- reproducible bring-up steps
- controller safety story with watchdogs
- honest handling of technical debt
- hardware/software co-design decisions

## Suggested Priority Order

1. Reproducibility
2. Cleanup of stale utilities and tests
3. Better demo media
4. More robust perception and telemetry
5. Larger architectural upgrades such as ROS 2 or SLAM
