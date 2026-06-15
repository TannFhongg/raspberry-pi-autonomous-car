"""
Fixed lane-width calibration for the current camera/config setup.

Usage:
    PYTHONPATH=. venv/bin/python calibrate_fixed.py test_full_hd.jpg

This script reads config/hardware_config.yaml, keeps the image at its native
resolution, and uses the same ROI/Canny/Hough settings as lane detection.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from perception.lane_detector import normalize_lane_config
from utils.config_loader import load_config


LANE_WIDTH_CM = 25.0
DEFAULT_CONFIG_PATH = "config/hardware_config.yaml"
DEFAULT_OUTPUT_PATH = "calibration_result_fixed.jpg"


def _extract_gray_frame(frame):
    """Return grayscale from BGR, grayscale, or YUV420-like input."""
    if frame is None or not hasattr(frame, "shape"):
        return None

    if len(frame.shape) == 2:
        height, width = frame.shape[:2]
        if height % 3 == 0 and width % 2 == 0:
            image_height = (height * 2) // 3
            return frame[:image_height, :].copy()
        return frame.copy()

    if len(frame.shape) == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return None


def _as_bgr_frame(frame):
    """Return a BGR frame for drawing."""
    if frame is None or not hasattr(frame, "shape"):
        return None

    if len(frame.shape) == 2:
        height, width = frame.shape[:2]
        if height % 3 == 0 and width % 2 == 0:
            return cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    if len(frame.shape) == 3 and frame.shape[2] == 3:
        return frame.copy()

    return None


def _calculate_x_at_bottom(lines, height, width):
    x_bottoms = []
    for x1, y1, x2, y2, slope in lines:
        x_bottom = x1 + (height - y1) / slope
        if 0 <= x_bottom <= width:
            x_bottoms.append(x_bottom)

    if not x_bottoms:
        return None
    return int(np.median(x_bottoms))


def _draw_text(frame, text, y, color=(255, 255, 255)):
    cv2.putText(
        frame,
        text,
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )


def calibrate_lane_width_fixed(
    frame,
    lane_config=None,
    output_path=DEFAULT_OUTPUT_PATH,
):
    """
    Measure lane width in pixels using the current lane detection config.

    Args:
        frame: Input image, preferably captured with the current camera setup.
        lane_config: ai.lane_detection config dictionary.
        output_path: Debug image path.

    Returns:
        Measured lane width in pixels, or None if calibration fails.
    """
    config = normalize_lane_config(lane_config)
    gray = _extract_gray_frame(frame)
    frame_calib = _as_bgr_frame(frame)

    if gray is None or frame_calib is None:
        print("❌ Frame không hợp lệ để calibration")
        return None

    height, width = gray.shape[:2]
    center_x = width // 2
    print(f"[INFO] Calibration resolution: {width}x{height}")
    print(
        "[INFO] Config: "
        f"ROI=({config['roi_left_ratio']:.2f},{config['roi_top_ratio']:.2f})"
        f"-({config['roi_right_ratio']:.2f},{config['roi_bottom_ratio']:.2f}), "
        f"Canny={config['canny_low']}/{config['canny_high']}, "
        f"Hough={config['hough_threshold']}, "
        f"minLen={config['min_line_length']}, maxGap={config['max_line_gap']}, "
        f"blur={config['blur_kernel']}"
    )

    gray_inv = cv2.bitwise_not(gray)
    blur = cv2.GaussianBlur(
        gray_inv,
        (config["blur_kernel"], config["blur_kernel"]),
        0,
    )
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(blur)
    edges = cv2.Canny(enhanced, config["canny_low"], config["canny_high"])

    roi_top = int(height * config["roi_top_ratio"])
    roi_bottom = int(height * config["roi_bottom_ratio"])
    roi_left = int(width * config["roi_left_ratio"])
    roi_right = int(width * config["roi_right_ratio"])
    roi_vertices = np.array([[
        (0, roi_bottom),
        (roi_left, roi_top),
        (roi_right, roi_top),
        (width, roi_bottom),
    ]], dtype=np.int32)

    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    cv2.polylines(frame_calib, roi_vertices, True, (255, 0, 0), 2)
    cv2.line(frame_calib, (center_x, 0), (center_x, height), (0, 255, 255), 2)

    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=config["hough_threshold"],
        minLineLength=config["min_line_length"],
        maxLineGap=config["max_line_gap"],
    )

    if lines is None:
        print("❌ Không tìm thấy đường thẳng nào trong ROI")
        cv2.imwrite(output_path, frame_calib)
        print(f"📸 Đã lưu ảnh debug: {output_path}")
        return None

    left_lines = []
    right_lines = []
    rejected_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        if abs(x2 - x1) < 1:
            rejected_lines.append((x1, y1, x2, y2))
            continue

        slope = (y2 - y1) / (x2 - x1)
        mid_x = (x1 + x2) / 2

        if slope < -0.5 and mid_x < center_x:
            left_lines.append((x1, y1, x2, y2, slope))
            cv2.line(frame_calib, (x1, y1), (x2, y2), (0, 255, 0), 3)
        elif slope > 0.5 and mid_x > center_x:
            right_lines.append((x1, y1, x2, y2, slope))
            cv2.line(frame_calib, (x1, y1), (x2, y2), (255, 0, 0), 3)
        else:
            rejected_lines.append((x1, y1, x2, y2))
            cv2.line(frame_calib, (x1, y1), (x2, y2), (80, 80, 80), 1)

    print(f"[INFO] Hough lines: {len(lines)}")
    print(
        f"[INFO] Left lines: {len(left_lines)}, "
        f"Right lines: {len(right_lines)}, Rejected: {len(rejected_lines)}"
    )

    left_x = _calculate_x_at_bottom(left_lines, height, width)
    right_x = _calculate_x_at_bottom(right_lines, height, width)

    if left_x is None or right_x is None:
        print("❌ Không tìm thấy đủ 2 vạch để đo")
        print(f"   Left X: {left_x}, Right X: {right_x}")
        print("💡 Kiểm tra ảnh debug:")
        print("   - ROI xanh dương có chứa cả 2 line không?")
        print("   - Line trái xanh lá và line phải xanh dương có đúng vạch thật không?")
        print("   - Nếu thấy quá nhiều line xám, tăng hough_threshold/min_line_length")
        cv2.imwrite(output_path, frame_calib)
        print(f"📸 Đã lưu ảnh debug: {output_path}")
        return None

    lane_width_pixels = right_x - left_x

    cv2.circle(frame_calib, (left_x, height - 12), 11, (0, 255, 0), -1)
    cv2.circle(frame_calib, (right_x, height - 12), 11, (255, 0, 0), -1)
    cv2.line(frame_calib, (left_x, height - 35), (right_x, height - 35), (0, 255, 255), 4)
    cv2.line(frame_calib, (left_x, height - 90), (left_x, height), (0, 255, 0), 2)
    cv2.line(frame_calib, (right_x, height - 90), (right_x, height), (255, 0, 0), 2)

    _draw_text(frame_calib, f"Lane: {lane_width_pixels}px = {LANE_WIDTH_CM:.0f}cm", 35, (0, 255, 255))
    _draw_text(frame_calib, f"Left: {left_x}px | Right: {right_x}px", 65)
    _draw_text(
        frame_calib,
        f"Hough L:{len(left_lines)} R:{len(right_lines)} rejected:{len(rejected_lines)}",
        95,
    )
    _draw_text(frame_calib, "Green=left | Blue=right | Gray=rejected", height - 15)

    cv2.imwrite(output_path, frame_calib)

    print(f"\n{'=' * 60}")
    print("✅ CALIBRATION THÀNH CÔNG:")
    print(f"  Lane Width (Real):  {LANE_WIDTH_CM:.0f} cm")
    print(f"  Lane Width (Pixel): {lane_width_pixels} px")
    print(f"  Scale Factor:       {LANE_WIDTH_CM / lane_width_pixels:.4f} cm/px")
    print(f"  Left X:             {left_x} px")
    print(f"  Right X:            {right_x} px")
    print(f"  Current config:     {config['lane_width_pixels']} px")
    print(f"  📸 Đã lưu:          {output_path}")
    print(f"{'=' * 60}\n")
    print("⚠️  CẬP NHẬT:")
    print("  config/hardware_config.yaml:")
    print(f"  ai.lane_detection.lane_width_pixels: {lane_width_pixels}")
    print(f"{'=' * 60}\n")

    return lane_width_pixels


def load_lane_config(config_path):
    try:
        config = load_config(config_path)
    except Exception as exc:
        print(f"⚠️  Không load được config {config_path}: {exc}")
        return None
    return config.get("ai", {}).get("lane_detection", {})


def main():
    parser = argparse.ArgumentParser(description="Calibrate lane_width_pixels from one image.")
    parser.add_argument("image", help="Input calibration image")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Hardware config YAML")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output debug image")
    args = parser.parse_args()

    image_path = Path(args.image)
    print(f"📸 Đọc ảnh: {image_path}")

    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"❌ Không đọc được ảnh: {image_path}")
        return 1

    print(f"   Kích thước gốc: {frame.shape[1]}x{frame.shape[0]}")
    lane_config = load_lane_config(args.config)
    result = calibrate_lane_width_fixed(
        frame,
        lane_config=lane_config,
        output_path=args.output,
    )

    if result is None:
        print("❌ Calibration thất bại. Xem ảnh debug và chỉnh ROI/Hough nếu cần.")
        return 1

    print(f"✅ Thành công! Lane width = {result} pixels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
