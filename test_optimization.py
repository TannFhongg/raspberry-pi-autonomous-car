#!/usr/bin/env python3
"""
Test Script - Kiểm tra hiệu năng sau tối ưu ISP
So sánh CPU load, FPS, latency trước và sau
"""

import cv2
import time
import numpy as np
import psutil
import os
from perception.camera_manager import CameraManager
from perception.lane_detector import detect_line
from utils.config_loader import load_config


def get_cpu_temp():
    """Đọc nhiệt độ CPU (Raspberry Pi)"""
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            temp = float(f.read()) / 1000.0
            return temp
    except:
        return None


def test_camera_performance(duration=10):
    """
    Test hiệu năng camera và lane detection
    
    Args:
        duration: Thời gian test (giây)
    """
    print("="*70)
    print("🚀 TEST HIỆU NĂNG SAU TỐI ƯU ISP")
    print("="*70)
    
    # Load config
    config = load_config()
    
    # Hiển thị cấu hình
    cam_config = config.get("sensors", {}).get("camera", {})
    resolution = cam_config.get("resolution", [640, 480])
    format_type = cam_config.get("picamera2", {}).get("format", "RGB888")
    
    print(f"\n📷 Cấu hình Camera:")
    print(f"  - Độ phân giải: {resolution[0]}x{resolution[1]}")
    print(f"  - Format: {format_type}")
    print(f"  - Framerate: {cam_config.get('framerate', 30)} fps")
    
    # Khởi tạo camera
    print(f"\n⏳ Khởi động camera...")
    camera = CameraManager(config)
    
    if not camera.start():
        print("❌ Không thể khởi động camera!")
        return
    
    print("✅ Camera đã sẵn sàng\n")
    
    # Đợi camera ổn định
    time.sleep(1.0)
    
    # Metrics
    frame_count = 0
    total_capture_time = 0
    total_detect_time = 0
    cpu_samples = []
    temp_samples = []
    
    start_time = time.time()
    last_print = start_time
    
    print(f"🔄 Bắt đầu test trong {duration} giây...\n")
    print(f"{'Time':<8} {'FPS':<8} {'CPU%':<8} {'Temp°C':<8} {'Capture':<10} {'Detect':<10}")
    print("-"*70)
    
    try:
        while time.time() - start_time < duration:
            # Đo thời gian capture
            t0 = time.time()
            frame = camera.capture_frame()
            t1 = time.time()
            
            if frame is None:
                continue
            
            capture_time = (t1 - t0) * 1000  # ms
            total_capture_time += capture_time
            
            # Đo thời gian lane detection
            t2 = time.time()
            error, x_line, center_x, frame_debug = detect_line(frame, config.get("ai", {}).get("lane_detection"))
            t3 = time.time()
            
            detect_time = (t3 - t2) * 1000  # ms
            total_detect_time += detect_time
            
            frame_count += 1
            
            # Lấy CPU và nhiệt độ
            cpu_percent = psutil.cpu_percent(interval=0)
            cpu_samples.append(cpu_percent)
            
            temp = get_cpu_temp()
            if temp:
                temp_samples.append(temp)
            
            # In thông tin mỗi giây
            current_time = time.time()
            if current_time - last_print >= 1.0:
                elapsed = current_time - start_time
                fps = frame_count / elapsed
                
                print(f"{elapsed:6.1f}s  {fps:6.1f}  {cpu_percent:6.1f}  "
                      f"{temp if temp else 'N/A':>6}  "
                      f"{capture_time:6.2f}ms  {detect_time:6.2f}ms")
                
                last_print = current_time
        
    except KeyboardInterrupt:
        print("\n⚠️  Test bị dừng bởi người dùng")
    
    finally:
        camera.stop()
    
    # Tính toán kết quả
    elapsed = time.time() - start_time
    avg_fps = frame_count / elapsed
    avg_capture = total_capture_time / frame_count
    avg_detect = total_detect_time / frame_count
    avg_total = avg_capture + avg_detect
    avg_cpu = np.mean(cpu_samples) if cpu_samples else 0
    avg_temp = np.mean(temp_samples) if temp_samples else None
    
    print("\n" + "="*70)
    print("📊 KẾT QUẢ TEST")
    print("="*70)
    
    print(f"\n⏱️  Thời gian:")
    print(f"  - Tổng thời gian test: {elapsed:.1f}s")
    print(f"  - Số frame xử lý: {frame_count}")
    
    print(f"\n🎬 FPS:")
    print(f"  - FPS trung bình: {avg_fps:.1f}")
    print(f"  - FPS mục tiêu: 30.0")
    if avg_fps >= 29.0:
        print(f"  ✅ Đạt mục tiêu!")
    else:
        print(f"  ⚠️  Chưa đạt mục tiêu")
    
    print(f"\n⚡ Latency:")
    print(f"  - Capture: {avg_capture:.2f}ms")
    print(f"  - Detection: {avg_detect:.2f}ms")
    print(f"  - Tổng: {avg_total:.2f}ms")
    print(f"  - Mục tiêu: <35ms")
    if avg_total < 35:
        print(f"  ✅ Đạt mục tiêu!")
    else:
        print(f"  ⚠️  Chưa đạt mục tiêu")
    
    print(f"\n💻 CPU:")
    print(f"  - CPU trung bình: {avg_cpu:.1f}%")
    print(f"  - CPU tối đa: {max(cpu_samples):.1f}%")
    print(f"  - CPU tối thiểu: {min(cpu_samples):.1f}%")
    print(f"  - Mục tiêu: <30%")
    if avg_cpu < 30:
        print(f"  ✅ Đạt mục tiêu!")
    else:
        print(f"  ⚠️  Chưa đạt mục tiêu")
    
    if avg_temp:
        print(f"\n🌡️  Nhiệt độ:")
        print(f"  - Nhiệt độ trung bình: {avg_temp:.1f}°C")
        print(f"  - Nhiệt độ tối đa: {max(temp_samples):.1f}°C")
        print(f"  - Nhiệt độ tối thiểu: {min(temp_samples):.1f}°C")
        print(f"  - Mục tiêu: <65°C")
        if avg_temp < 65:
            print(f"  ✅ Đạt mục tiêu!")
        else:
            print(f"  ⚠️  Chưa đạt mục tiêu")
    
    # Phân tích chi tiết
    print(f"\n📈 Phân tích:")
    
    # Băng thông
    if format_type == "YUV420":
        bytes_per_frame = resolution[0] * resolution[1] * 1.5
        print(f"  - Format: YUV420 (Tối ưu)")
    elif format_type == "RGB888":
        bytes_per_frame = resolution[0] * resolution[1] * 3
        print(f"  - Format: RGB888 (Chưa tối ưu)")
    else:
        bytes_per_frame = 0
    
    if bytes_per_frame > 0:
        bandwidth_mbps = (bytes_per_frame * avg_fps) / (1024 * 1024)
        print(f"  - Băng thông: {bandwidth_mbps:.2f} MB/s")
    
    # So sánh với baseline
    print(f"\n📊 So sánh với baseline (RGB888 1640x1232):")
    baseline_cpu = 45
    baseline_fps = 26
    baseline_temp = 68
    
    cpu_improvement = ((baseline_cpu - avg_cpu) / baseline_cpu) * 100
    fps_improvement = ((avg_fps - baseline_fps) / baseline_fps) * 100
    
    print(f"  - CPU: {avg_cpu:.1f}% (giảm {cpu_improvement:.1f}%)")
    print(f"  - FPS: {avg_fps:.1f} (tăng {fps_improvement:.1f}%)")
    
    if avg_temp:
        temp_improvement = ((baseline_temp - avg_temp) / baseline_temp) * 100
        print(f"  - Nhiệt độ: {avg_temp:.1f}°C (giảm {temp_improvement:.1f}%)")
    
    # Kết luận
    print(f"\n🎯 Kết luận:")
    
    success_count = 0
    total_checks = 3
    
    if avg_fps >= 29.0:
        success_count += 1
    if avg_cpu < 30:
        success_count += 1
    if avg_temp and avg_temp < 65:
        success_count += 1
    
    if success_count == total_checks:
        print(f"  ✅ Tối ưu THÀNH CÔNG! ({success_count}/{total_checks} mục tiêu đạt)")
        print(f"  🚀 Hệ thống hoạt động mượt mà và hiệu quả!")
    elif success_count >= 2:
        print(f"  ⚠️  Tối ưu TỐT! ({success_count}/{total_checks} mục tiêu đạt)")
        print(f"  💡 Có thể cải thiện thêm một số điểm")
    else:
        print(f"  ❌ Cần tối ưu thêm! ({success_count}/{total_checks} mục tiêu đạt)")
        print(f"  🔧 Kiểm tra lại cấu hình và phần cứng")
    
    print("\n" + "="*70)


def compare_formats():
    """
    So sánh hiệu năng giữa các format khác nhau
    (Chỉ chạy được nếu có quyền thay đổi config)
    """
    print("\n🔬 So sánh các format:")
    print("  - YUV420: Tối ưu cho Grayscale")
    print("  - RGB888: Cần cho màu sắc")
    print("  - MJPEG: Tối ưu cho streaming")
    print("\n💡 Để test, thay đổi 'format' trong config/hardware_config.yaml")


if __name__ == "__main__":
    import sys
    
    # Kiểm tra tham số
    duration = 10
    if len(sys.argv) > 1:
        try:
            duration = int(sys.argv[1])
        except:
            print("Usage: python3 test_optimization.py [duration_seconds]")
            sys.exit(1)
    
    # Chạy test
    test_camera_performance(duration)
    
    # Hiển thị thông tin thêm
    compare_formats()
    
    print("\n💡 Gợi ý:")
    print("  - Chạy 'htop' để xem chi tiết CPU usage")
    print("  - Chạy 'vcgencmd measure_temp' để xem nhiệt độ realtime")
    print("  - Chạy 'python3 test_lane_optimized.py' để test lane detection")
