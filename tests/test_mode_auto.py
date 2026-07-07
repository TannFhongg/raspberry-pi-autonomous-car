#!/usr/bin/env python3
"""
Auto mode sign-size tuning dashboard.

Run on the robot:
    python test_mode_auto.py

Open:
    http://<raspberry-pi-ip>:5002

This tool does not drive the motors. It only measures the same sign metric used
by AutoModeController:

    sign_size = max(bounding_box_width, bounding_box_height)
"""

from __future__ import annotations

import argparse
import atexit
import re
import sys
import threading
import time
from collections import deque
from pathlib import Path
from statistics import median
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
CONTROLLER_PATH = ROOT / "control" / "robot_controller.py"

SIGN_CLASSES = [
    "left_turn_sign",
    "right_turn_sign",
    "stop_sign",
    "red_light",
    "green_light",
    "speed_limit_signs",
    "parking_signs",
]

TARGET_OPTIONS = ["auto_largest", "any_sign", *SIGN_CLASSES]

DEFAULT_SETTINGS = {
    "target_sign": "any_sign",
    "prepare_size": 150,
    "execute_size": 250,
    "conf_threshold": 0.5,
}

ZONE_COLORS = {
    "SEARCH": (100, 116, 139),
    "TOO FAR": (245, 158, 11),
    "PREPARE": (37, 99, 235),
    "EXECUTE": (220, 38, 38),
}

app = Flask(__name__)
tuner: "AutoModeSignTuner | None" = None


def clamp_int(value: Any, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(round(float(value)))))


def clamp_float(value: Any, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def load_initial_settings(config_path: Path, controller_path: Path) -> dict[str, Any]:
    settings = DEFAULT_SETTINGS.copy()
    config_loaded = False

    try:
        config = load_config(str(config_path))
        lane_config = config.get("lane_following", {})
        sign_config = lane_config.get("sign_detection", {})
        if "dist_prepare" in sign_config or "dist_prepare" in lane_config:
            config_loaded = True
        settings["prepare_size"] = int(
            sign_config.get(
                "dist_prepare",
                lane_config.get("dist_prepare", settings["prepare_size"]),
            )
        )
        if "dist_execute" in sign_config or "dist_execute" in lane_config:
            config_loaded = True
        settings["execute_size"] = int(
            sign_config.get(
                "dist_execute",
                lane_config.get("dist_execute", settings["execute_size"]),
            )
        )
    except Exception as exc:
        print(f"Could not load sign thresholds from {config_path}: {exc}")

    if not config_loaded:
        try:
            text = controller_path.read_text(encoding="utf-8")
            prepare_match = re.search(r"self\.DIST_PREPARE\s*=\s*(\d+)", text)
            execute_match = re.search(r"self\.DIST_EXECUTE\s*=\s*(\d+)", text)
            if prepare_match:
                settings["prepare_size"] = int(prepare_match.group(1))
            if execute_match:
                settings["execute_size"] = int(execute_match.group(1))
        except Exception as exc:
            print(f"Could not read {controller_path}: {exc}")

    if settings["execute_size"] <= settings["prepare_size"]:
        settings["execute_size"] = settings["prepare_size"] + 1

    return settings


def format_yaml_scalar(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def save_auto_sign_params_to_config(
    config_path: Path,
    prepare_size: int,
    execute_size: int,
) -> None:
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    values = {
        "dist_prepare": int(prepare_size),
        "dist_execute": int(execute_size),
    }

    lines = config_path.read_text(encoding="utf-8").splitlines()

    lane_start = None
    for index, line in enumerate(lines):
        if line.strip().startswith("#") or not line.strip():
            continue
        if line.startswith("lane_following:"):
            lane_start = index
            break

    if lane_start is None:
        lines.extend(["", "lane_following:"])
        lane_start = len(lines) - 1

    lane_end = len(lines)
    for index in range(lane_start + 1, len(lines)):
        line = lines[index]
        if line.strip() and not line.startswith(" "):
            lane_end = index
            break

    sign_start = None
    sign_end = lane_end
    for index in range(lane_start + 1, lane_end):
        if lines[index].startswith("  sign_detection:"):
            sign_start = index
            break

    if sign_start is None:
        insert_at = lane_end
        block = [
            "",
            "  # Sign detection thresholds - đo bằng test_mode_auto.py, metric max(width, height)",
            "  sign_detection:",
            f"    dist_prepare: {format_yaml_scalar(values['dist_prepare'])}",
            f"    dist_execute: {format_yaml_scalar(values['dist_execute'])}",
        ]
        lines[insert_at:insert_at] = block
        config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    for index in range(sign_start + 1, lane_end):
        line = lines[index]
        if line.strip() and line.startswith("  ") and not line.startswith("    "):
            sign_end = index
            break

    updated = set()
    for index in range(sign_start + 1, sign_end):
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
        insert_at = sign_start + 1
        insert_lines = [f"    {name}: {format_yaml_scalar(values[name])}" for name in missing]
        lines[insert_at:insert_at] = insert_lines

    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_samples(samples: deque[dict[str, Any]]) -> dict[str, int]:
    sizes = [int(item.get("size", 0)) for item in samples if int(item.get("size", 0)) > 0]
    averages = [int(item.get("average", 0)) for item in samples if int(item.get("average", 0)) > 0]
    values = averages or sizes

    if not values:
        return {"count": 0, "mean": 0, "median": 0, "min": 0, "max": 0}

    return {
        "count": len(values),
        "mean": int(round(sum(values) / len(values))),
        "median": int(round(median(values))),
        "min": min(values),
        "max": max(values),
    }


class AutoModeSignTuner:
    def __init__(self, config_path: Path, model_path: Path, controller_path: Path):
        self.config_path = config_path
        self.model_path = model_path
        self.controller_path = controller_path
        self.config = self._load_config()

        self.settings_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        self.settings = load_initial_settings(config_path, controller_path)

        self.size_history: deque[int] = deque(maxlen=30)
        self.samples: deque[dict[str, Any]] = deque(maxlen=30)
        self.prepare_samples: deque[dict[str, Any]] = deque(maxlen=20)
        self.execute_samples: deque[dict[str, Any]] = deque(maxlen=20)

        self.status: dict[str, Any] = self._initial_status()
        self.latest_jpeg = self._encode_placeholder("Starting camera...")

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
            "target_sign": self.settings["target_sign"],
            "sign_name": "",
            "zone": "SEARCH",
            "current_size": 0,
            "average_size": 0,
            "prepare_size": self.settings["prepare_size"],
            "execute_size": self.settings["execute_size"],
            "to_prepare": 0,
            "to_execute": 0,
            "target_x": 0,
            "target_y": 0,
            "target_w": 0,
            "target_h": 0,
            "confidence": 0,
            "fps": 0.0,
            "frame_width": 0,
            "frame_height": 0,
            "samples": [],
            "prepare_samples": [],
            "execute_samples": [],
            "prepare_summary": summarize_samples(self.prepare_samples),
            "execute_summary": summarize_samples(self.execute_samples),
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
            if "target_sign" in payload:
                target_sign = str(payload["target_sign"])
                if target_sign not in TARGET_OPTIONS:
                    raise ValueError("Invalid sign target")
                self.settings["target_sign"] = target_sign
                self.size_history.clear()

            if "prepare_size" in payload:
                self.settings["prepare_size"] = clamp_int(payload["prepare_size"], 20, 1200)

            if "execute_size" in payload:
                self.settings["execute_size"] = clamp_int(payload["execute_size"], 21, 1400)

            if self.settings["execute_size"] <= self.settings["prepare_size"]:
                self.settings["execute_size"] = self.settings["prepare_size"] + 1

            if "conf_threshold" in payload:
                self.settings["conf_threshold"] = round(
                    clamp_float(payload["conf_threshold"], 0.05, 0.95), 2
                )

            return self.settings.copy()

    def use_current_size(self, threshold: str, source: str = "average") -> dict[str, Any]:
        with self.frame_lock:
            current = int(self.status.get("current_size", 0) or 0)
            average = int(round(self.status.get("average_size", 0) or 0))

        size = average if source == "average" and average > 0 else current
        if size <= 0:
            raise ValueError("No sign is being tracked")

        if threshold == "prepare":
            return self.update_settings({"prepare_size": size})
        if threshold == "execute":
            return self.update_settings({"execute_size": size})
        raise ValueError("Invalid threshold")

    def snapshot(self, kind: str = "sample") -> dict[str, Any]:
        with self.frame_lock:
            if not self.status.get("tracking"):
                raise ValueError("No sign is being tracked")

            sample = {
                "time": time.strftime("%H:%M:%S"),
                "kind": kind,
                "sign": str(self.status.get("sign_name", "")),
                "zone": str(self.status.get("zone", "SEARCH")),
                "size": int(self.status.get("current_size", 0) or 0),
                "average": int(round(self.status.get("average_size", 0) or 0)),
                "confidence": int(self.status.get("confidence", 0) or 0),
            }

            if kind == "prepare":
                self.prepare_samples.appendleft(sample)
            elif kind == "execute":
                self.execute_samples.appendleft(sample)
            else:
                self.samples.appendleft(sample)

            self._publish_sample_lists_locked()
            return sample

    def reset_samples(self) -> None:
        with self.frame_lock:
            self.samples.clear()
            self.prepare_samples.clear()
            self.execute_samples.clear()
            self.size_history.clear()
            self._publish_sample_lists_locked()
            self.status["average_size"] = 0

    def save_current_settings(self) -> None:
        settings = self.get_settings()
        save_auto_sign_params_to_config(
            self.config_path,
            int(settings["prepare_size"]),
            int(settings["execute_size"]),
        )

    def get_status(self) -> dict[str, Any]:
        with self.frame_lock:
            status = self.status.copy()
            status["samples"] = list(self.samples)
            status["prepare_samples"] = list(self.prepare_samples)
            status["execute_samples"] = list(self.execute_samples)
            status["prepare_summary"] = summarize_samples(self.prepare_samples)
            status["execute_summary"] = summarize_samples(self.execute_samples)

        settings = self.get_settings()
        status.update(settings)

        current_size = int(status.get("current_size", 0) or 0)
        prepare_size = int(settings["prepare_size"])
        execute_size = int(settings["execute_size"])
        status["to_prepare"] = prepare_size - current_size
        status["to_execute"] = execute_size - current_size

        if current_size > 0:
            if current_size < prepare_size:
                status["zone"] = "TOO FAR"
            elif current_size < execute_size:
                status["zone"] = "PREPARE"
            else:
                status["zone"] = "EXECUTE"

        return status

    def get_jpeg(self) -> bytes:
        with self.frame_lock:
            return bytes(self.latest_jpeg)

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

            frame_raw = self.camera.capture_frame()
            if frame_raw is None:
                self._publish_placeholder("Waiting for camera frame...")
                time.sleep(0.1)
                continue

            frame_bgr = frame_to_bgr(frame_raw, self.camera.format)

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
                self._publish_sample_lists_locked()

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
        target_sign = settings["target_sign"]
        prepare_size = int(settings["prepare_size"])
        execute_size = int(settings["execute_size"])
        conf_threshold = float(settings["conf_threshold"])

        valid = [det for det in detections if float(det.get("conf", 0.0)) >= conf_threshold]

        if target_sign == "auto_largest":
            matching = valid
        elif target_sign == "any_sign":
            matching = [det for det in valid if det.get("class_name") in SIGN_CLASSES]
        else:
            matching = [det for det in valid if det.get("class_name") == target_sign]

        target = max(matching, key=lambda item: item["w"] * item["h"]) if matching else None

        cv2.line(frame, (frame_w // 2, 0), (frame_w // 2, frame_h), (148, 163, 184), 1)
        cv2.line(frame, (0, frame_h // 2), (frame_w, frame_h // 2), (148, 163, 184), 1)

        stats = {
            "tracking": False,
            "message": f"Searching {target_sign}",
            "target_sign": target_sign,
            "sign_name": "",
            "zone": "SEARCH",
            "current_size": 0,
            "average_size": int(round(np.mean(self.size_history))) if self.size_history else 0,
            "prepare_size": prepare_size,
            "execute_size": execute_size,
            "to_prepare": 0,
            "to_execute": 0,
            "target_x": 0,
            "target_y": 0,
            "target_w": 0,
            "target_h": 0,
            "confidence": 0,
            "frame_width": frame_w,
            "frame_height": frame_h,
        }

        for det in matching:
            self._draw_detection(frame, det, (148, 163, 184), thickness=1, show_label=True)

        if target:
            obj_size = int(round(max(float(target["w"]), float(target["h"]))))
            if obj_size < prepare_size:
                zone = "TOO FAR"
                message = "Sign seen, still before PREPARE"
            elif obj_size < execute_size:
                zone = "PREPARE"
                message = "At PREPARE distance"
            else:
                zone = "EXECUTE"
                message = "At EXECUTE distance"

            self.size_history.append(obj_size)
            average_size = int(round(np.mean(self.size_history)))
            color = ZONE_COLORS[zone]

            self._draw_detection(frame, target, color, thickness=3, show_label=False)
            self._draw_target_label(frame, target, obj_size, average_size, zone, color)

            stats.update(
                {
                    "tracking": True,
                    "message": message,
                    "sign_name": str(target["class_name"]),
                    "zone": zone,
                    "current_size": obj_size,
                    "average_size": average_size,
                    "to_prepare": prepare_size - obj_size,
                    "to_execute": execute_size - obj_size,
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
        show_label: bool,
    ) -> None:
        x, y, w, h = (float(det["x"]), float(det["y"]), float(det["w"]), float(det["h"]))
        x1 = max(0, int(round(x - w / 2)))
        y1 = max(0, int(round(y - h / 2)))
        x2 = min(frame.shape[1] - 1, int(round(x + w / 2)))
        y2 = min(frame.shape[0] - 1, int(round(y + h / 2)))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        if show_label:
            label = f"{det['class_name']} {int(float(det['conf']) * 100)}%"
            cv2.putText(
                frame,
                label,
                (x1 + 4, max(18, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    def _draw_target_label(
        self,
        frame: np.ndarray,
        det: dict[str, Any],
        obj_size: int,
        average_size: int,
        zone: str,
        color: tuple[int, int, int],
    ) -> None:
        x, y, w, h = (float(det["x"]), float(det["y"]), float(det["w"]), float(det["h"]))
        x1 = max(0, int(round(x - w / 2)))
        y1 = max(32, int(round(y - h / 2)))
        label = (
            f"{det['class_name']} {int(float(det['conf']) * 100)}% | "
            f"size {obj_size}px | avg {average_size}px | {zone}"
        )
        cv2.rectangle(frame, (x1, y1 - 28), (min(frame.shape[1] - 1, x1 + 620), y1), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 8, y1 - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def _draw_header(
        self,
        frame: np.ndarray,
        stats: dict[str, Any],
        settings: dict[str, Any],
        fps: float,
    ) -> None:
        prepare = int(settings["prepare_size"])
        execute = int(settings["execute_size"])
        lines = [
            f"Auto sign tuner | prepare {prepare}px | execute {execute}px | measured=max(w,h)",
            f"{stats['zone']} | sign {stats['sign_name'] or '-'} | current {stats['current_size']}px | avg {stats['average_size']}px | fps {fps:.1f}",
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
                    "zone": "SEARCH",
                    "current_size": 0,
                    "confidence": 0,
                }
            )

    def _publish_sample_lists_locked(self) -> None:
        self.status["samples"] = list(self.samples)
        self.status["prepare_samples"] = list(self.prepare_samples)
        self.status["execute_samples"] = list(self.execute_samples)
        self.status["prepare_summary"] = summarize_samples(self.prepare_samples)
        self.status["execute_summary"] = summarize_samples(self.execute_samples)


HTML_TEMPLATE = """
<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Auto Mode Sign Tuner</title>
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
      grid-template-columns: minmax(0, 1.45fr) minmax(360px, 0.65fr);
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
      grid-template-columns: minmax(126px, 0.75fr) minmax(0, 1fr) 76px;
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
      grid-template-columns: repeat(2, minmax(0, 1fr));
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
    button.amber { background: var(--amber); }
    button.red { background: var(--red); }
    button:active { transform: translateY(1px); }

    .summary-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }

    .summary-box {
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 10px;
      background: #fff;
      display: grid;
      gap: 4px;
      min-height: 88px;
    }

    .summary-box label {
      color: var(--muted);
      font-size: 12px;
    }

    .summary-box strong {
      font-size: 24px;
      line-height: 1;
    }

    .summary-box span {
      color: var(--muted);
      font-size: 12px;
    }

    .sample-list {
      display: grid;
      gap: 6px;
      max-height: 245px;
      overflow: auto;
    }

    .sample {
      display: grid;
      grid-template-columns: 58px 74px 1fr 72px;
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
      .button-row, .summary-grid { grid-template-columns: 1fr; }
      .sample { grid-template-columns: 58px 1fr 72px; }
      .sample span:nth-child(3) { display: none; }
    }
  </style>
</head>
<body>
  <header class="topbar">
    <div class="title">
      <h1>Auto Mode Sign Tuner</h1>
      <span>Measure AutoModeController sign_size = max(width, height), without moving motors.</span>
    </div>
    <div class="state-pill"><span id="ready-dot" class="dot"></span><span id="ready-text">Starting</span></div>
  </header>

  <main>
    <section class="panel">
      <div class="panel-header">
        <h2>Camera</h2>
        <div class="state-pill"><span id="target-sign">any_sign</span></div>
      </div>
      <div class="video-wrap">
        <img id="camera-feed" src="/video_feed" alt="Camera feed">
      </div>
      <div class="metrics">
        <div class="metric">
          <label>Current size</label>
          <strong id="current-size">0</strong>
          <span>px, max(w,h)</span>
        </div>
        <div class="metric">
          <label>Average</label>
          <strong id="average-size">0</strong>
          <span>last 30 frames</span>
        </div>
        <div class="metric">
          <label>To PREPARE</label>
          <strong id="to-prepare">0</strong>
          <span>prepare - current</span>
        </div>
        <div class="metric">
          <label>To EXECUTE</label>
          <strong id="to-execute">0</strong>
          <span>execute - current</span>
        </div>
      </div>
      <div class="decision">
        <div>
          <small>Auto zone</small><br>
          <strong id="zone">SEARCH</strong>
        </div>
        <div>
          <small>Detected sign</small><br>
          <strong id="sign-name">-</strong>
        </div>
        <div>
          <small>Thresholds</small><br>
          <strong id="thresholds">150 / 250 px</strong>
        </div>
      </div>
    </section>

    <aside class="panel">
      <div class="panel-header">
        <h2>Measure</h2>
        <div class="state-pill"><span id="fps">0.0 FPS</span></div>
      </div>
      <div class="controls">
        <div class="control-group">
          <div class="control-row">
            <label for="target-sign-select">Target sign</label>
            <select id="target-sign-select">
              <option value="auto_largest">auto_largest</option>
              <option value="any_sign">any_sign</option>
              <option value="left_turn_sign">left_turn_sign</option>
              <option value="right_turn_sign">right_turn_sign</option>
              <option value="stop_sign">stop_sign</option>
              <option value="red_light">red_light</option>
              <option value="green_light">green_light</option>
              <option value="speed_limit_signs">speed_limit_signs</option>
              <option value="parking_signs">parking_signs</option>
            </select>
            <span></span>
          </div>
          <div class="control-row">
            <label for="prepare-range">Prepare px</label>
            <input id="prepare-range" type="range" min="20" max="1200" step="1">
            <input id="prepare-number" type="number" min="20" max="1200" step="1">
          </div>
          <div class="control-row">
            <label for="execute-range">Execute px</label>
            <input id="execute-range" type="range" min="21" max="1400" step="1">
            <input id="execute-number" type="number" min="21" max="1400" step="1">
          </div>
          <div class="control-row">
            <label for="conf-range">Min confidence</label>
            <input id="conf-range" type="range" min="0.05" max="0.95" step="0.01">
            <input id="conf-number" type="number" min="0.05" max="0.95" step="0.01">
          </div>
        </div>

        <div class="button-row">
          <button id="mark-prepare" class="amber">Mark PREPARE</button>
          <button id="mark-execute" class="red">Mark EXECUTE</button>
          <button id="use-prepare" class="secondary">Use avg as PREPARE</button>
          <button id="use-execute" class="secondary">Use avg as EXECUTE</button>
          <button id="snapshot" class="green">Snapshot</button>
          <button id="reset-samples" class="secondary">Reset samples</button>
          <button id="save-config">Save config</button>
        </div>

        <div class="summary-grid">
          <div class="summary-box">
            <label>Suggested PREPARE</label>
            <strong id="prepare-median">0</strong>
            <span id="prepare-summary">0 samples</span>
          </div>
          <div class="summary-box">
            <label>Suggested EXECUTE</label>
            <strong id="execute-median">0</strong>
            <span id="execute-summary">0 samples</span>
          </div>
        </div>

        <div class="control-group">
          <label>Recent samples</label>
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
      targetSign: el("target-sign-select"),
      prepareRange: el("prepare-range"),
      prepareNumber: el("prepare-number"),
      executeRange: el("execute-range"),
      executeNumber: el("execute-number"),
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
      fields.targetSign.value = settings.target_sign;
      fields.prepareRange.value = settings.prepare_size;
      fields.prepareNumber.value = settings.prepare_size;
      fields.executeRange.value = settings.execute_size;
      fields.executeNumber.value = settings.execute_size;
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
          syncControls(data.settings);
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

    fields.targetSign.addEventListener("change", () => {
      if (!state.applying) updateSettings({ target_sign: fields.targetSign.value });
    });
    bindPair(fields.prepareRange, fields.prepareNumber, "prepare_size", (value) => parseInt(value, 10));
    bindPair(fields.executeRange, fields.executeNumber, "execute_size", (value) => parseInt(value, 10));
    bindPair(fields.confRange, fields.confNumber, "conf_threshold", Number);

    async function postJson(path, payload = {}) {
      const response = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.message || "Request failed");
      return data;
    }

    async function mark(kind) {
      try {
        const data = await postJson("/snapshot", { kind });
        toast(`Marked ${kind.toUpperCase()}: ${data.sample.average || data.sample.size}px`);
        renderSamples(data.samples || []);
      } catch (error) {
        toast(error.message, true);
      }
    }

    el("mark-prepare").addEventListener("click", () => mark("prepare"));
    el("mark-execute").addEventListener("click", () => mark("execute"));
    el("snapshot").addEventListener("click", () => mark("sample"));

    async function useCurrent(threshold) {
      try {
        const data = await postJson("/use_current", { threshold, source: "average" });
        state.settings = data.settings;
        syncControls(data.settings);
        toast(`${threshold.toUpperCase()} = ${data.settings[threshold + "_size"]}px`);
      } catch (error) {
        toast(error.message, true);
      }
    }

    el("use-prepare").addEventListener("click", () => useCurrent("prepare"));
    el("use-execute").addEventListener("click", () => useCurrent("execute"));

    el("reset-samples").addEventListener("click", async () => {
      await postJson("/reset_samples");
      renderSamples([]);
      toast("Samples reset");
    });

    el("save-config").addEventListener("click", async () => {
      try {
        await postJson("/save_config");
        toast("Saved to hardware_config.yaml");
      } catch (error) {
        toast(error.message, true);
      }
    });

    function renderSamples(samples) {
      const root = el("samples");
      if (!samples.length) {
        root.innerHTML = `<div class="sample"><span>--:--</span><span>No sample</span><span></span><span></span></div>`;
        return;
      }
      root.innerHTML = samples.map((sample) => `
        <div class="sample">
          <span>${sample.time}</span>
          <span>${sample.kind}</span>
          <span>${sample.sign} / ${sample.zone}</span>
          <span>${sample.average || sample.size}px</span>
        </div>
      `).join("");
    }

    function paintSummary(prefix, summary) {
      el(`${prefix}-median`).textContent = summary.median || 0;
      el(`${prefix}-summary`).textContent =
        `${summary.count || 0} samples | avg ${summary.mean || 0}px | ${summary.min || 0}-${summary.max || 0}px`;
    }

    function paintStatus(data) {
      el("ready-dot").classList.toggle("on", Boolean(data.ready));
      el("ready-text").textContent = data.ready ? "Camera ready" : data.message;
      el("target-sign").textContent = data.target_sign;
      el("current-size").textContent = data.current_size;
      el("average-size").textContent = data.average_size;
      el("to-prepare").textContent = data.to_prepare;
      el("to-execute").textContent = data.to_execute;
      el("zone").textContent = data.zone;
      el("sign-name").textContent = data.sign_name || "-";
      el("thresholds").textContent = `${data.prepare_size} / ${data.execute_size} px`;
      el("fps").textContent = `${data.fps.toFixed(1)} FPS`;
      paintSummary("prepare", data.prepare_summary || {});
      paintSummary("execute", data.execute_summary || {});
      renderSamples([...(data.prepare_samples || []), ...(data.execute_samples || []), ...(data.samples || [])].slice(0, 18));
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
        updated = tuner.use_current_size(
            threshold=str(payload.get("threshold", "prepare")),
            source=str(payload.get("source", "average")),
        )
        return jsonify({"status": "success", "settings": updated})
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


@app.route("/snapshot", methods=["POST"])
def snapshot():
    if tuner is None:
        return jsonify({"status": "error", "message": "Tuner is not initialized"}), 503
    try:
        payload = request.get_json(silent=True) or {}
        kind = str(payload.get("kind", "sample"))
        if kind not in {"sample", "prepare", "execute"}:
            raise ValueError("Invalid sample kind")
        sample = tuner.snapshot(kind)
        return jsonify(
            {
                "status": "success",
                "sample": sample,
                "samples": [
                    *list(tuner.prepare_samples),
                    *list(tuner.execute_samples),
                    *list(tuner.samples),
                ],
            }
        )
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
    parser = argparse.ArgumentParser(description="Auto mode sign-size tuning UI")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5002)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--controller", default=str(CONTROLLER_PATH))
    return parser.parse_args()


def main() -> None:
    global tuner

    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    controller_path = Path(args.controller).expanduser().resolve()

    tuner = AutoModeSignTuner(
        config_path=config_path,
        model_path=model_path,
        controller_path=controller_path,
    )
    tuner.start()
    atexit.register(tuner.stop)

    print(f"Auto mode sign tuner: http://{args.host}:{args.port}")
    print(f"Config:     {config_path}")
    print(f"Model:      {model_path}")
    print(f"Controller: {controller_path}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
