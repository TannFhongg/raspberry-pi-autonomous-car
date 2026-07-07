# API

## Scope

This repository contains multiple Flask-based web surfaces:

- the main robot runtime in [`main.py`](../main.py)
- the lane tuning dashboard in [`dashboard_server.py`](../dashboard_server.py)
- the follow-target tuner in [`test_follow.py`](../test_follow.py)
- the auto sign-size tuner in [`test_mode_auto.py`](../test_mode_auto.py)

This document lists only endpoints that actually exist in those files.

## 1. Main Runtime API (`main.py`)

Base URL:

- `http://<robot-ip>:5000`

### HTTP endpoints

| Endpoint | Method | Purpose | Parameters | Example | Expected behavior |
| --- | --- | --- | --- | --- | --- |
| `/` | `GET` | Render the main dashboard | none | `GET /` | Returns the HTML UI |
| `/video_feed` | `GET` | MJPEG stream from the shared camera | none | `GET /video_feed` | Returns multipart JPEG stream |
| `/debug_feed` | `GET` | MJPEG debug/monitor stream | none | `GET /debug_feed` | Shows current mode-specific debug feed |
| `/set_mode` | `GET` | Switch runtime mode | `mode=auto|follow|idle` | `GET /set_mode?mode=auto` | Starts target mode or returns error if unavailable |
| `/set_follow_color` | `GET` | Select follow-mode target color | `color=red|green|blue|yellow` | `GET /set_follow_color?color=red` | Updates follow controller target class |
| `/stop` | `GET` | Stop active controllers and return to idle | none | `GET /stop` | Stops motors/controllers and emits idle mode |
| `/emergency_stop` | `GET` | Immediate emergency stop | none | `GET /emergency_stop` | Stops motors and marks controller state as emergency |
| `/set_speed` | `GET` | Set runtime speed cap | `value=0..255` | `GET /set_speed?value=120` | Updates robot speed and propagates to active mode controllers |
| `/read_log` | `GET` | Read the runtime log file | none | `GET /read_log` | Returns plain-text log contents |
| `/clear_log` | `GET` | Clear the runtime log file | none | `GET /clear_log` | Rewrites the log file with a “cleared” entry |

### Mode notes

- Valid runtime modes are `auto`, `follow`, and `idle`.
- Manual directional routes such as `/forward` and `/left` do not exist in the current main runtime.

### Example calls

```bash
curl "http://<robot-ip>:5000/set_mode?mode=auto"
curl "http://<robot-ip>:5000/set_speed?value=100"
curl "http://<robot-ip>:5000/set_follow_color?color=yellow"
curl "http://<robot-ip>:5000/stop"
```

## 2. Real-Time Socket.IO Events (`main.py`)

The main dashboard also relies on Socket.IO.

### Server-emitted events

| Event | Payload | Purpose |
| --- | --- | --- |
| `connection_response` | `{ "data": "Connected" }` | Confirms the client is connected |
| `mode_update` | `{ "mode": "auto" }` | Pushes current mode to the frontend |
| `sensor_update` | runtime state dictionary | Pushes speed, state, lane position, etc. |
| `target_update` | follow target dictionary | Pushes follow-mode tracking data |
| `log_entry` | `{time, level, message}` | Streams log lines to the UI |
| `arduino_sensors` | Arduino sensor/status dictionary | Forwards messages coming from the Arduino driver |

### Client-originated events

The current code explicitly handles:

- `connect`
- `disconnect`

There are no custom command Socket.IO events in the current runtime; control is done through HTTP routes.

## 3. Lane Tuning Dashboard API (`dashboard_server.py`)

Base URL:

- `http://<robot-ip>:5001`

Important limitation:

- this dashboard is a tuning/visualization surface
- its `/start` and `/stop` routes only change dashboard status text, not the real robot runtime in `main.py`

| Endpoint | Method | Purpose | Parameters | Example | Expected behavior |
| --- | --- | --- | --- | --- | --- |
| `/` | `GET` | Render the lane tuning dashboard | none | `GET /` | Returns HTML tuning UI |
| `/video_feed` | `GET` | MJPEG debug feed with lane overlays | none | `GET /video_feed` | Streams annotated frames |
| `/stats` | `GET` | Read current lane status | none | `GET /stats` | Returns `error`, `lane_status`, dashboard `robot_status`, and `speed` |
| `/get_params` | `GET` | Read current tuning parameters | none | `GET /get_params` | Returns current lane tuning dict |
| `/update_param` | `POST` | Update one tuning parameter | JSON `{name, value}` | `POST /update_param` | Validates and stores one slider value |
| `/save_params` | `POST` | Persist lane params to YAML | none | `POST /save_params` | Writes current lane params back into `hardware_config.yaml` |
| `/start` | `POST` | Mark dashboard state as running | none | `POST /start` | Sets local dashboard status to `RUNNING` |
| `/stop` | `POST` | Mark dashboard state as stopped | none | `POST /stop` | Sets local dashboard status to `STOPPED` |

### Example JSON body

```json
{
  "name": "blur_kernel",
  "value": 9
}
```

## 4. Follow Target Tuner API (`test_follow.py`)

Default base URL:

- `http://<robot-ip>:5001`

Practical note:

- change the port if `dashboard_server.py` is also running, because both default to `5001`

| Endpoint | Method | Purpose | Parameters | Example | Expected behavior |
| --- | --- | --- | --- | --- | --- |
| `/` | `GET` | Render the follow target tuner | none | `GET /` | Returns HTML tuning UI |
| `/video_feed` | `GET` | MJPEG stream with target overlays | none | `GET /video_feed` | Streams annotated frames |
| `/status` | `GET` | Read current target-tracking state | none | `GET /status` | Returns target size, band, decision, FPS, and samples |
| `/settings` | `POST` | Update follow tuning settings | JSON partial settings | `POST /settings` | Validates and stores target size, tolerance, color, or confidence |
| `/use_current` | `POST` | Set target size from current measurement | optional JSON | `POST /use_current` | Copies the current or average measured target size into settings |
| `/snapshot` | `POST` | Save one measurement sample | none | `POST /snapshot` | Appends a target-size sample if tracking is active |
| `/reset_samples` | `POST` | Clear saved samples | none | `POST /reset_samples` | Empties sample history |
| `/save_config` | `POST` | Persist tuner settings | none | `POST /save_config` | Writes `follow_mode` settings to YAML |

### Example settings payload

```json
{
  "target_color": "green",
  "target_size": 220,
  "size_tolerance": 30,
  "conf_threshold": 0.6
}
```

## 5. Auto Sign-Size Tuner API (`test_mode_auto.py`)

Base URL:

- `http://<robot-ip>:5002`

| Endpoint | Method | Purpose | Parameters | Example | Expected behavior |
| --- | --- | --- | --- | --- | --- |
| `/` | `GET` | Render the sign-size tuning UI | none | `GET /` | Returns HTML tuning UI |
| `/video_feed` | `GET` | MJPEG stream with sign overlays | none | `GET /video_feed` | Streams annotated frames |
| `/status` | `GET` | Read current sign-size status | none | `GET /status` | Returns current size, average, zone, and saved samples |
| `/settings` | `POST` | Update sign tuning settings | JSON partial settings | `POST /settings` | Updates target sign, thresholds, or confidence |
| `/use_current` | `POST` | Copy current measurement into one threshold | JSON `{threshold, source}` | `POST /use_current` | Writes measured size into `prepare` or `execute` |
| `/snapshot` | `POST` | Save a sign-size sample | JSON `{kind}` | `POST /snapshot` | Saves a general, `prepare`, or `execute` sample |
| `/reset_samples` | `POST` | Clear sample history | none | `POST /reset_samples` | Clears sign tuning samples |
| `/save_config` | `POST` | Persist sign thresholds | none | `POST /save_config` | Writes values into `lane_following.sign_detection` |

### Example payloads

```json
{
  "target_sign": "left_turn_sign",
  "prepare_size": 150,
  "execute_size": 250,
  "conf_threshold": 0.5
}
```

```json
{
  "threshold": "prepare",
  "source": "average"
}
```

## 6. Error Handling Notes

Common patterns used by these APIs:

- `400` for invalid parameters or malformed JSON
- `503` when a tuner/controller is not initialized
- `500` for file write or runtime failures

Operational caveats:

- many state-changing routes use `GET` rather than `POST` in the main runtime
- the tuning dashboards are not secured or authenticated
- multiple dashboards can fight for the same camera resource if started together
