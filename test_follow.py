#!/usr/bin/env python3
"""
Follow mode target-size tuning dashboard.

Run on the robot:
    python test_follow.py

Open:
    http://<raspberry-pi-ip>:5001
"""

from __future__ import annotations

import argparse
import atexit
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from perception.camera_manager import frame_to_bgr, get_web_camera, release_web_camera
from perception.object_detector import ObjectDetector
from utils.config_loader import load_config


CONFIG_PATH = ROOT / "config" / "hardware_config.yaml"
MODEL_PATH = ROOT / "models" / "best_ncnn_model"

COLOR_MAP = {
    "red": "red_color",
    "green": "green_color",
    "blue": "blue_color",
    "yellow": "yellow_color",
}

UI_COLOR = {
    "red": "#dc2626",
    "green": "#059669",
    "blue": "#2563eb",
    "yellow": "#d97706",
}

DEFAULT_SETTINGS = {
    "target_color": "red",
    "target_size": 350,
    "size_tolerance": 20,
    "conf_threshold": 0.5,
}

app = Flask(__name__)
tuner: "FollowTargetTuner | None" = None


def load_initial_settings(config_path: Path) -> dict[str, Any]:
    settings = DEFAULT_SETTINGS.copy()
    try:
        config = load_config(str(config_path))
        follow_config = config.get("follow_mode", {}) or {}
        settings["target_size"] = int(follow_config.get("target_size", settings["target_size"]))
        settings["size_tolerance"] = int(
            follow_config.get("size_tolerance", settings["size_tolerance"])
        )
    except Exception as exc:
        print(f"Could not load {config_path}: {exc}")
    return settings


def clamp_int(value: Any, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(round(float(value)))))


def clamp_float(value: Any, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def format_yaml_scalar(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def save_follow_params_to_config(config_path: Path, target_size: int, tolerance: int) -> None:
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    values = {
        "target_size": int(target_size),
        "size_tolerance": int(tolerance),
    }

    lines = config_path.read_text(encoding="utf-8").splitlines()

    section_start = None
    for index, line in enumerate(lines):
        if line.strip().startswith("#") or not line.strip():
            continue
        if line.startswith("follow_mode:"):
            section_start = index
            break

    if section_start is None:
        lines.extend(["", "follow_mode:"])
        section_start = len(lines) - 1

    section_end = len(lines)
    for index in range(section_start + 1, len(lines)):
        line = lines[index]
        if line.strip() and not line.startswith(" ") and not line.strip().startswith("#"):
            section_end = index
            break

    updated = set()
    for index in range(section_start + 1, section_end):
        line = lines[index]
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue

        key = stripped.split(":", 1)[0].strip()
        if key not in values:
            continue

        before_comment, separator, comment = line.partition("#")
        prefix = before_comment.split(":", 1)[0] + ": "
        comment_padding = before_comment[len(before_comment.rstrip()) :]
        suffix = f"{comment_padding}#{comment}" if separator else ""
        lines[index] = prefix + format_yaml_scalar(values[key]) + suffix
        updated.add(key)

    missing = [name for name in values if name not in updated]
    if missing:
        insert_at = section_start + 1
        insert_lines = [f"  {name}: {format_yaml_scalar(values[name])}" for name in missing]
        lines[insert_at:insert_at] = insert_lines

    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class FollowTargetTuner:
    def __init__(self, config_path: Path, model_path: Path):
        self.config_path = config_path
        self.model_path = model_path
        self.config = self._load_config()
        self.settings_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        self.settings = load_initial_settings(config_path)
        self.status: dict[str, Any] = self._initial_status()
        self.latest_jpeg = self._encode_placeholder("Starting camera...")
        self.samples: deque[dict[str, Any]] = deque(maxlen=20)
        self.size_history: deque[int] = deque(maxlen=30)
        self.running = False
        self.thread: threading.Thread | None = None
        self.camera = None
        self.detector: ObjectDetector | None = None

    def _load_config(self) -> dict[str, Any]:
        try:
            return load_config(str(self.config_path))
        except Exception as exc:
            print(f"Could not load config, using defaults: {exc}")
            return {}

    def _initial_status(self) -> dict[str, Any]:
        return {
            "ready": False,
            "tracking": False,
            "message": "Starting",
            "target_class": COLOR_MAP[self.settings["target_color"]],
            "current_size": 0,
            "average_size": 0,
            "target_size": self.settings["target_size"],
            "size_tolerance": self.settings["size_tolerance"],
            "size_min": self.settings["target_size"] - self.settings["size_tolerance"],
            "size_max": self.settings["target_size"] + self.settings["size_tolerance"],
            "size_error": 0,
            "decision": "SEARCH",
            "horizontal_error": 0,
            "target_x": 0,
            "target_y": 0,
            "target_w": 0,
            "target_h": 0,
            "confidence": 0,
            "fps": 0.0,
            "frame_width": 0,
            "frame_height": 0,
            "sample_count": 0,
            "samples": [],
        }

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        release_web_camera()

    def get_settings(self) -> dict[str, Any]:
        with self.settings_lock:
            return self.settings.copy()

    def update_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.settings_lock:
            if "target_color" in payload:
                color = str(payload["target_color"])
                if color not in COLOR_MAP:
                    raise ValueError("Invalid target color")
                self.settings["target_color"] = color

            if "target_size" in payload:
                self.settings["target_size"] = clamp_int(payload["target_size"], 50, 1200)

            if "size_tolerance" in payload:
                self.settings["size_tolerance"] = clamp_int(payload["size_tolerance"], 0, 300)

            if "conf_threshold" in payload:
                self.settings["conf_threshold"] = round(
                    clamp_float(payload["conf_threshold"], 0.05, 0.95), 2
                )

            return self.settings.copy()

    def use_current_size(self, source: str = "average") -> dict[str, Any]:
        with self.frame_lock:
            current = int(self.status.get("current_size", 0) or 0)
            average = int(round(self.status.get("average_size", 0) or 0))

        size = average if source == "average" and average > 0 else current
        if size <= 0:
            raise ValueError("No target is being tracked")
        return self.update_settings({"target_size": size})

    def snapshot(self) -> dict[str, Any]:
        with self.frame_lock:
            if not self.status.get("tracking"):
                raise ValueError("No target is being tracked")
            sample = {
                "time": time.strftime("%H:%M:%S"),
                "size": int(self.status.get("current_size", 0)),
                "average": int(round(self.status.get("average_size", 0) or 0)),
                "error": int(self.status.get("size_error", 0)),
                "confidence": int(self.status.get("confidence", 0)),
            }
            self.samples.appendleft(sample)
            self.status["samples"] = list(self.samples)
            self.status["sample_count"] = len(self.samples)
            return sample

    def reset_samples(self) -> None:
        with self.frame_lock:
            self.samples.clear()
            self.size_history.clear()
            self.status["samples"] = []
            self.status["sample_count"] = 0
            self.status["average_size"] = 0

    def get_status(self) -> dict[str, Any]:
        with self.frame_lock:
            status = self.status.copy()
            status["samples"] = list(self.samples)
        settings = self.get_settings()
        status.update(settings)
        status["target_class"] = COLOR_MAP[status["target_color"]]
        status["color_hex"] = UI_COLOR[status["target_color"]]
        status["size_min"] = int(settings["target_size"]) - int(settings["size_tolerance"])
        status["size_max"] = int(settings["target_size"]) + int(settings["size_tolerance"])

        current_size = int(status.get("current_size", 0) or 0)
        if current_size > 0:
            status["size_error"] = int(settings["target_size"]) - current_size
            if status["size_min"] <= current_size <= status["size_max"]:
                status["decision"] = "LOCKED"
            elif current_size < status["size_min"]:
                status["decision"] = "TOO FAR - FORWARD"
            else:
                status["decision"] = "TOO CLOSE - BACKWARD"

        return status

    def get_jpeg(self) -> bytes:
        with self.frame_lock:
            return bytes(self.latest_jpeg)

    def save_current_settings(self) -> None:
        settings = self.get_settings()
        save_follow_params_to_config(
            self.config_path,
            int(settings["target_size"]),
            int(settings["size_tolerance"]),
        )

    def _ensure_detector(self) -> bool:
        if self.detector is not None:
            return self.detector.model is not None

        try:
            self.detector = ObjectDetector(
                model_path=str(self.model_path),
                conf_threshold=float(self.settings["conf_threshold"]),
            )
            return self.detector.model is not None
        except Exception as exc:
            self._publish_placeholder(f"Model error: {exc}")
            return False

    def _ensure_camera(self) -> bool:
        try:
            if self.camera is None:
                self.camera = get_web_camera(self.config)
            if not self.camera.is_running():
                return bool(self.camera.start())
            return True
        except Exception as exc:
            self._publish_placeholder(f"Camera error: {exc}")
            return False

    def _loop(self) -> None:
        last_fps_time = time.time()
        frame_counter = 0
        fps = 0.0

        while self.running:
            if not self._ensure_camera() or not self._ensure_detector():
                time.sleep(1.0)
                continue

            frame_yuv = self.camera.capture_frame()
            if frame_yuv is None:
                self._publish_placeholder("Waiting for camera frame...")
                time.sleep(0.1)
                continue

            frame_bgr = frame_to_bgr(frame_yuv, self.camera.format)

            settings = self.get_settings()
            if self.detector:
                self.detector.conf_threshold = float(settings["conf_threshold"])

            detections, _ = self.detector.detect(frame_bgr, draw_boxes=False)
            annotated, stats = self._annotate(frame_bgr, detections, settings, fps)
            jpeg = self._encode_frame(annotated)

            frame_counter += 1
            now = time.time()
            if now - last_fps_time >= 1.0:
                fps = frame_counter / (now - last_fps_time)
                frame_counter = 0
                last_fps_time = now

            with self.frame_lock:
                self.latest_jpeg = jpeg
                self.status.update(stats)
                self.status["fps"] = round(fps, 1)
                self.status["ready"] = True
                self.status["samples"] = list(self.samples)
                self.status["sample_count"] = len(self.samples)

            time.sleep(0.03)

    def _annotate(
        self,
        frame_bgr: np.ndarray,
        detections: list[dict[str, Any]],
        settings: dict[str, Any],
        fps: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        frame = frame_bgr.copy()
        frame_h, frame_w = frame.shape[:2]
        target_class = COLOR_MAP[settings["target_color"]]
        target_size = int(settings["target_size"])
        tolerance = int(settings["size_tolerance"])
        size_min = target_size - tolerance
        size_max = target_size + tolerance

        matching = [
            det
            for det in detections
            if det.get("class_name") == target_class
            and float(det.get("conf", 0.0)) >= float(settings["conf_threshold"])
        ]

        target = max(matching, key=lambda item: item["w"] * item["h"]) if matching else None

        cv2.line(frame, (frame_w // 2, 0), (frame_w // 2, frame_h), (148, 163, 184), 1)
        cv2.line(frame, (0, frame_h // 2), (frame_w, frame_h // 2), (148, 163, 184), 1)

        stats = {
            "tracking": False,
            "message": f"Searching {target_class}",
            "target_class": target_class,
            "current_size": 0,
            "target_size": target_size,
            "size_tolerance": tolerance,
            "size_min": size_min,
            "size_max": size_max,
            "size_error": 0,
            "decision": "SEARCH",
            "horizontal_error": 0,
            "target_x": 0,
            "target_y": 0,
            "target_w": 0,
            "target_h": 0,
            "confidence": 0,
            "frame_width": frame_w,
            "frame_height": frame_h,
            "average_size": int(round(np.mean(self.size_history))) if self.size_history else 0,
        }

        for det in matching:
            self._draw_detection(frame, det, (148, 163, 184), thickness=1)

        if target:
            obj_size = int(round(max(float(target["w"]), float(target["h"]))))
            error = target_size - obj_size
            horizontal_error = int(round(float(target["x"]) - (frame_w / 2)))

            if size_min <= obj_size <= size_max:
                decision = "LOCKED"
                color = (22, 163, 74)
                message = f"Locked on {target_class}"
            elif obj_size < size_min:
                decision = "TOO FAR - FORWARD"
                color = (245, 158, 11)
                message = f"Too far from {target_class}"
            else:
                decision = "TOO CLOSE - BACKWARD"
                color = (220, 38, 38)
                message = f"Too close to {target_class}"

            self.size_history.append(obj_size)
            average_size = int(round(np.mean(self.size_history)))
            self._draw_detection(frame, target, color, thickness=3)
            self._draw_target_label(frame, target, obj_size, average_size, error, color)

            stats.update(
                {
                    "tracking": True,
                    "message": message,
                    "current_size": obj_size,
                    "average_size": average_size,
                    "size_error": error,
                    "decision": decision,
                    "horizontal_error": horizontal_error,
                    "target_x": int(round(float(target["x"]))),
                    "target_y": int(round(float(target["y"]))),
                    "target_w": int(round(float(target["w"]))),
                    "target_h": int(round(float(target["h"]))),
                    "confidence": int(round(float(target["conf"]) * 100)),
                }
            )

        self._draw_header(frame, stats, settings, fps)
        return frame, stats

    def _draw_detection(
        self,
        frame: np.ndarray,
        det: dict[str, Any],
        color: tuple[int, int, int],
        thickness: int,
    ) -> None:
        x, y, w, h = (float(det["x"]), float(det["y"]), float(det["w"]), float(det["h"]))
        x1 = max(0, int(round(x - w / 2)))
        y1 = max(0, int(round(y - h / 2)))
        x2 = min(frame.shape[1] - 1, int(round(x + w / 2)))
        y2 = min(frame.shape[0] - 1, int(round(y + h / 2)))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    def _draw_target_label(
        self,
        frame: np.ndarray,
        det: dict[str, Any],
        obj_size: int,
        average_size: int,
        error: int,
        color: tuple[int, int, int],
    ) -> None:
        x, y, w, h = (float(det["x"]), float(det["y"]), float(det["w"]), float(det["h"]))
        x1 = max(0, int(round(x - w / 2)))
        y1 = max(28, int(round(y - h / 2)))
        label = f"{det['class_name']} {int(det['conf'] * 100)}% | size {obj_size}px | avg {average_size}px | err {error:+d}"
        cv2.rectangle(frame, (x1, y1 - 26), (min(frame.shape[1] - 1, x1 + 520), y1), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 8, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def _draw_header(self, frame: np.ndarray, stats: dict[str, Any], settings: dict[str, Any], fps: float) -> None:
        target = int(settings["target_size"])
        tolerance = int(settings["size_tolerance"])
        lines = [
            f"Follow target tuner | target {target}px +/- {tolerance}px | measured=max(w,h)",
            f"{stats['decision']} | current {stats['current_size']}px | avg {stats['average_size']}px | fps {fps:.1f}",
        ]
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 62), (17, 24, 39), -1)
        for index, text in enumerate(lines):
            cv2.putText(
                frame,
                text,
                (14, 24 + index * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def _encode_frame(self, frame: np.ndarray) -> bytes:
        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            return self._encode_placeholder("JPEG encode failed")
        return buffer.tobytes()

    def _encode_placeholder(self, message: str) -> bytes:
        frame = np.full((540, 960, 3), (246, 247, 251), dtype=np.uint8)
        cv2.putText(
            frame,
            message[:80],
            (40, 270),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (31, 41, 55),
            2,
            cv2.LINE_AA,
        )
        return self._encode_frame(frame)

    def _publish_placeholder(self, message: str) -> None:
        with self.frame_lock:
            self.latest_jpeg = self._encode_placeholder(message)
            self.status.update(
                {
                    "ready": False,
                    "tracking": False,
                    "message": message,
                    "decision": "WAITING",
                    "current_size": 0,
                    "confidence": 0,
                }
            )


HTML_TEMPLATE = """
<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Follow Target Tuner</title>
  <style>
    :root {
      --bg: #f6f7fb;
      --panel: #ffffff;
      --ink: #111827;
      --muted: #667085;
      --line: #d9dee8;
      --accent: #2563eb;
      --green: #059669;
      --amber: #d97706;
      --red: #dc2626;
      --shadow: 0 10px 28px rgba(17, 24, 39, 0.08);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: Inter, "Segoe UI", Roboto, Arial, sans-serif;
      color: var(--ink);
      background:
        linear-gradient(180deg, rgba(255,255,255,0.92), rgba(246,247,251,0.96)),
        repeating-linear-gradient(90deg, rgba(37,99,235,0.05) 0, rgba(37,99,235,0.05) 1px, transparent 1px, transparent 80px);
    }

    .topbar {
      min-height: 58px;
      padding: 12px clamp(14px, 2vw, 28px);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.9);
      backdrop-filter: blur(10px);
      position: sticky;
      top: 0;
      z-index: 10;
    }

    .title {
      display: flex;
      flex-direction: column;
      gap: 2px;
      min-width: 0;
    }

    .title h1 {
      margin: 0;
      font-size: clamp(18px, 2vw, 24px);
      line-height: 1.15;
      letter-spacing: 0;
    }

    .title span {
      color: var(--muted);
      font-size: 13px;
    }

    .state-pill {
      min-height: 32px;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 10px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fff;
      color: var(--muted);
      white-space: nowrap;
      font-size: 13px;
      font-weight: 700;
    }

    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--amber);
    }

    .dot.on { background: var(--green); }

    main {
      width: min(100%, 1680px);
      margin: 0 auto;
      padding: clamp(14px, 2vw, 24px);
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) minmax(340px, 0.65fr);
      gap: clamp(14px, 1.5vw, 22px);
      align-items: start;
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    .panel-header {
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }

    .panel-header h2 {
      margin: 0;
      font-size: 15px;
      letter-spacing: 0;
    }

    .video-wrap {
      background: #111827;
      aspect-ratio: 4 / 3;
      display: grid;
      place-items: center;
    }

    #camera-feed {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
    }

    .metrics {
      padding: 12px;
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      border-top: 1px solid var(--line);
    }

    .metric {
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 10px;
      min-height: 74px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      gap: 6px;
      background: #fbfcff;
    }

    .metric label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }

    .metric strong {
      font-size: clamp(20px, 2vw, 28px);
      line-height: 1;
    }

    .metric span {
      color: var(--muted);
      font-size: 12px;
    }

    .controls {
      padding: 12px;
      display: grid;
      gap: 12px;
    }

    .control-group {
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 12px;
      display: grid;
      gap: 10px;
      background: #fbfcff;
    }

    .control-row {
      display: grid;
      grid-template-columns: minmax(120px, 0.75fr) minmax(0, 1fr) 76px;
      gap: 10px;
      align-items: center;
    }

    label {
      color: #344054;
      font-size: 13px;
      font-weight: 700;
    }

    input[type="range"] { width: 100%; }

    input[type="number"], select {
      width: 100%;
      height: 36px;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      padding: 0 9px;
      background: white;
      color: var(--ink);
      font: inherit;
    }

    .button-row {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }

    button {
      min-height: 38px;
      border: 1px solid transparent;
      border-radius: 6px;
      padding: 8px 10px;
      font-weight: 800;
      font-size: 13px;
      cursor: pointer;
      color: white;
      background: var(--accent);
    }

    button.secondary {
      background: white;
      color: var(--ink);
      border-color: #cbd5e1;
    }

    button.green { background: var(--green); }
    button.red { background: var(--red); }

    button:active { transform: translateY(1px); }

    .decision {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 12px;
      border-top: 1px solid var(--line);
      background: #fbfcff;
    }

    .decision strong {
      font-size: 18px;
      line-height: 1.2;
    }

    .decision small {
      color: var(--muted);
      font-weight: 700;
    }

    .sample-list {
      display: grid;
      gap: 6px;
      max-height: 250px;
      overflow: auto;
    }

    .sample {
      display: grid;
      grid-template-columns: 68px 1fr 1fr 1fr;
      gap: 8px;
      align-items: center;
      padding: 7px 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      font-size: 12px;
    }

    .toast {
      position: fixed;
      bottom: 18px;
      left: 50%;
      transform: translateX(-50%);
      min-width: min(420px, calc(100vw - 32px));
      padding: 12px 14px;
      border-radius: 7px;
      background: #111827;
      color: #fff;
      box-shadow: var(--shadow);
      opacity: 0;
      pointer-events: none;
      transition: opacity 160ms ease;
      z-index: 100;
      text-align: center;
      font-weight: 700;
    }

    .toast.show { opacity: 1; }
    .toast.error { background: var(--red); }

    @media (max-width: 1040px) {
      main { grid-template-columns: 1fr; }
      .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }

    @media (max-width: 620px) {
      .topbar { align-items: flex-start; flex-direction: column; }
      .metrics { grid-template-columns: 1fr; }
      .control-row { grid-template-columns: 1fr; }
      .button-row { grid-template-columns: 1fr; }
      .sample { grid-template-columns: 58px 1fr 1fr; }
      .sample span:last-child { display: none; }
    }
  </style>
</head>
<body>
  <header class="topbar">
    <div class="title">
      <h1>Follow Target Tuner</h1>
      <span>Đo size theo đúng follow mode: max(width, height) của bounding box.</span>
    </div>
    <div class="state-pill"><span id="ready-dot" class="dot"></span><span id="ready-text">Starting</span></div>
  </header>

  <main>
    <section class="panel">
      <div class="panel-header">
        <h2>Camera</h2>
        <div class="state-pill"><span id="target-class">red_color</span></div>
      </div>
      <div class="video-wrap">
        <img id="camera-feed" src="/video_feed" alt="Camera feed">
      </div>
      <div class="metrics">
        <div class="metric">
          <label>Size hiện tại</label>
          <strong id="current-size">0</strong>
          <span>px</span>
        </div>
        <div class="metric">
          <label>Trung bình</label>
          <strong id="average-size">0</strong>
          <span>30 frame gần nhất</span>
        </div>
        <div class="metric">
          <label>Sai số</label>
          <strong id="size-error">0</strong>
          <span>target - current</span>
        </div>
        <div class="metric">
          <label>Confidence</label>
          <strong id="confidence">0</strong>
          <span>%</span>
        </div>
      </div>
      <div class="decision">
        <div>
          <small>Quyết định giả lập</small><br>
          <strong id="decision">SEARCH</strong>
        </div>
        <div>
          <small>Target band</small><br>
          <strong id="target-band">330 - 370 px</strong>
        </div>
      </div>
    </section>

    <aside class="panel">
      <div class="panel-header">
        <h2>Điều chỉnh</h2>
        <div class="state-pill"><span id="fps">0.0 FPS</span></div>
      </div>
      <div class="controls">
        <div class="control-group">
          <div class="control-row">
            <label for="target-color">Màu target</label>
            <select id="target-color">
              <option value="red">red_color</option>
              <option value="green">green_color</option>
              <option value="blue">blue_color</option>
              <option value="yellow">yellow_color</option>
            </select>
            <span></span>
          </div>
          <div class="control-row">
            <label for="target-size-range">Target size</label>
            <input id="target-size-range" type="range" min="50" max="1200" step="1">
            <input id="target-size-number" type="number" min="50" max="1200" step="1">
          </div>
          <div class="control-row">
            <label for="tolerance-range">Tolerance</label>
            <input id="tolerance-range" type="range" min="0" max="300" step="1">
            <input id="tolerance-number" type="number" min="0" max="300" step="1">
          </div>
          <div class="control-row">
            <label for="conf-range">Min confidence</label>
            <input id="conf-range" type="range" min="0.05" max="0.95" step="0.01">
            <input id="conf-number" type="number" min="0.05" max="0.95" step="0.01">
          </div>
        </div>

        <div class="button-row">
          <button id="use-current" class="green">Dùng size hiện tại</button>
          <button id="snapshot" class="secondary">Ghi mẫu</button>
          <button id="save-config">Lưu config</button>
        </div>
        <button id="reset-samples" class="secondary">Xóa mẫu đo</button>

        <div class="control-group">
          <label>Mẫu đo</label>
          <div id="samples" class="sample-list"></div>
        </div>
      </div>
    </aside>
  </main>

  <div id="toast" class="toast"></div>

  <script>
    const initialSettings = {{ settings_json | safe }};
    const state = { settings: initialSettings, applying: false, timer: null };

    const el = (id) => document.getElementById(id);
    const fields = {
      color: el("target-color"),
      targetRange: el("target-size-range"),
      targetNumber: el("target-size-number"),
      toleranceRange: el("tolerance-range"),
      toleranceNumber: el("tolerance-number"),
      confRange: el("conf-range"),
      confNumber: el("conf-number"),
    };

    function toast(message, error = false) {
      const box = el("toast");
      box.textContent = message;
      box.classList.toggle("error", error);
      box.classList.add("show");
      setTimeout(() => box.classList.remove("show"), 1800);
    }

    function syncControls(settings) {
      state.applying = true;
      fields.color.value = settings.target_color;
      fields.targetRange.value = settings.target_size;
      fields.targetNumber.value = settings.target_size;
      fields.toleranceRange.value = settings.size_tolerance;
      fields.toleranceNumber.value = settings.size_tolerance;
      fields.confRange.value = settings.conf_threshold;
      fields.confNumber.value = settings.conf_threshold;
      state.applying = false;
    }

    function updateSettings(partial) {
      clearTimeout(state.timer);
      state.timer = setTimeout(async () => {
        try {
          const response = await fetch("/settings", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(partial),
          });
          const data = await response.json();
          if (!response.ok) throw new Error(data.message || "Update failed");
          state.settings = data.settings;
        } catch (error) {
          toast(error.message, true);
        }
      }, 80);
    }

    function bindPair(range, number, key, parser = Number) {
      const handle = (source, target) => {
        if (state.applying) return;
        target.value = source.value;
        updateSettings({ [key]: parser(source.value) });
      };
      range.addEventListener("input", () => handle(range, number));
      number.addEventListener("input", () => handle(number, range));
    }

    fields.color.addEventListener("change", () => {
      if (!state.applying) updateSettings({ target_color: fields.color.value });
    });
    bindPair(fields.targetRange, fields.targetNumber, "target_size", (value) => parseInt(value, 10));
    bindPair(fields.toleranceRange, fields.toleranceNumber, "size_tolerance", (value) => parseInt(value, 10));
    bindPair(fields.confRange, fields.confNumber, "conf_threshold", Number);

    el("use-current").addEventListener("click", async () => {
      try {
        const response = await fetch("/use_current", { method: "POST" });
        const data = await response.json();
        if (!response.ok) throw new Error(data.message || "No target");
        state.settings = data.settings;
        syncControls(data.settings);
        toast(`Target size = ${data.settings.target_size}px`);
      } catch (error) {
        toast(error.message, true);
      }
    });

    el("snapshot").addEventListener("click", async () => {
      try {
        const response = await fetch("/snapshot", { method: "POST" });
        const data = await response.json();
        if (!response.ok) throw new Error(data.message || "No target");
        toast(`Đã ghi mẫu ${data.sample.size}px`);
        renderSamples(data.samples || []);
      } catch (error) {
        toast(error.message, true);
      }
    });

    el("reset-samples").addEventListener("click", async () => {
      await fetch("/reset_samples", { method: "POST" });
      renderSamples([]);
      toast("Đã xóa mẫu đo");
    });

    el("save-config").addEventListener("click", async () => {
      try {
        const response = await fetch("/save_config", { method: "POST" });
        const data = await response.json();
        if (!response.ok) throw new Error(data.message || "Save failed");
        toast("Đã lưu vào hardware_config.yaml");
      } catch (error) {
        toast(error.message, true);
      }
    });

    function renderSamples(samples) {
      const root = el("samples");
      if (!samples.length) {
        root.innerHTML = `<div class="sample"><span>--:--</span><span>Chưa có mẫu</span><span></span><span></span></div>`;
        return;
      }
      root.innerHTML = samples.map((sample) => `
        <div class="sample">
          <span>${sample.time}</span>
          <span>${sample.size}px</span>
          <span>avg ${sample.average}px</span>
          <span>err ${sample.error}</span>
        </div>
      `).join("");
    }

    function paintStatus(data) {
      el("ready-dot").classList.toggle("on", Boolean(data.ready));
      el("ready-text").textContent = data.ready ? "Camera ready" : data.message;
      el("target-class").textContent = data.target_class;
      el("current-size").textContent = data.current_size;
      el("average-size").textContent = data.average_size;
      el("size-error").textContent = data.size_error;
      el("confidence").textContent = data.confidence;
      el("decision").textContent = data.decision;
      el("target-band").textContent = `${data.size_min} - ${data.size_max} px`;
      el("fps").textContent = `${data.fps.toFixed(1)} FPS`;
      renderSamples(data.samples || []);
    }

    async function poll() {
      try {
        const response = await fetch("/status", { cache: "no-store" });
        const data = await response.json();
        paintStatus(data);
      } catch (error) {
        el("ready-text").textContent = "Disconnected";
        el("ready-dot").classList.remove("on");
      }
    }

    syncControls(initialSettings);
    renderSamples([]);
    poll();
    setInterval(poll, 350);
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    if tuner is None:
        return "Tuner is not initialized", 503
    import json

    return render_template_string(
        HTML_TEMPLATE,
        settings_json=json.dumps(tuner.get_settings()),
    )


@app.route("/video_feed")
def video_feed():
    if tuner is None:
        return "Tuner is not initialized", 503

    def frames():
        while True:
            frame = tuner.get_jpeg()
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.05)

    return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    if tuner is None:
        return jsonify({"status": "error", "message": "Tuner is not initialized"}), 503
    return jsonify(tuner.get_status())


@app.route("/settings", methods=["POST"])
def settings():
    if tuner is None:
        return jsonify({"status": "error", "message": "Tuner is not initialized"}), 503
    try:
        updated = tuner.update_settings(request.get_json(silent=True) or {})
        return jsonify({"status": "success", "settings": updated})
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


@app.route("/use_current", methods=["POST"])
def use_current():
    if tuner is None:
        return jsonify({"status": "error", "message": "Tuner is not initialized"}), 503
    try:
        payload = request.get_json(silent=True) or {}
        updated = tuner.use_current_size(str(payload.get("source", "average")))
        return jsonify({"status": "success", "settings": updated})
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


@app.route("/snapshot", methods=["POST"])
def snapshot():
    if tuner is None:
        return jsonify({"status": "error", "message": "Tuner is not initialized"}), 503
    try:
        sample = tuner.snapshot()
        return jsonify({"status": "success", "sample": sample, "samples": list(tuner.samples)})
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


@app.route("/reset_samples", methods=["POST"])
def reset_samples():
    if tuner is None:
        return jsonify({"status": "error", "message": "Tuner is not initialized"}), 503
    tuner.reset_samples()
    return jsonify({"status": "success"})


@app.route("/save_config", methods=["POST"])
def save_config():
    if tuner is None:
        return jsonify({"status": "error", "message": "Tuner is not initialized"}), 503
    try:
        tuner.save_current_settings()
        return jsonify({"status": "success", "config_path": str(tuner.config_path)})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Follow mode target-size tuning UI")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--model", default=str(MODEL_PATH))
    return parser.parse_args()


def main() -> None:
    global tuner

    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()

    tuner = FollowTargetTuner(config_path=config_path, model_path=model_path)
    tuner.start()
    atexit.register(tuner.stop)

    print(f"Follow target tuner: http://{args.host}:{args.port}")
    print(f"Config: {config_path}")
    print(f"Model:  {model_path}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
