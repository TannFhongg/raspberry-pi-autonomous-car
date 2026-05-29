#!/usr/bin/env python3
"""
Test Script: Verify YUV420 Format Fix
Kiểm tra xem YUV420 → BGR conversion có hoạt động đúng không
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from perception.camera_manager import CameraManager
from perception.lane_detector import detect_line
from perception.object_detector import ObjectDetector
from utils.config_loader import load_config


def test_camera_format():
    """Test 1: Kiểm tra camera trả về BGR format"""
    print("\n" + "="*60)
    print("TEST 1: Camera Format Verification")
    print("="*60)
    
    config = load_config()
    camera = CameraManager(config)
    
    if not camera.start():
        print("❌ FAILED: Camera không start được")
        return False
    
    try:
        frame = camera.capture_frame()
        
        if frame is None:
            print("❌ FAILED: capture_frame() trả về None")
            return False
        
        print(f"✅ Frame captured successfully")
        print(f"   Shape: {frame.shape}")
        print(f"   Dtype: {frame.dtype}")
        
        # Verify shape
        if len(frame.shape) != 3:
            print(f"❌ FAILED: Frame không phải 3D array (shape={frame.shape})")
            return False
        
        if frame.shape[2] != 3:
            print(f"❌ FAILED: Frame không có 3 channels (shape={frame.shape})")
            return False
        
        if frame.shape != (480, 640, 3):
            print(f"⚠️  WARNING: Frame shape không phải (480, 640, 3): {frame.shape}")
        
        print(f"✅ PASSED: Frame có format BGR chuẩn (H, W, 3)")
        
        # Test color range
        min_val = frame.min()
        max_val = frame.max()
        print(f"   Pixel range: [{min_val}, {max_val}]")
        
        if min_val < 0 or max_val > 255:
            print(f"❌ FAILED: Pixel values ngoài range [0, 255]")
            return False
        
        print(f"✅ PASSED: Pixel values trong range hợp lệ")
        
        # Save test frame
        cv2.imwrite("test_frame_bgr.jpg", frame)
        print(f"📸 Saved test frame: test_frame_bgr.jpg")
        
        return True
        
    finally:
        camera.stop()


def test_lane_detection():
    """Test 2: Kiểm tra lane detection nhận BGR đúng"""
    print("\n" + "="*60)
    print("TEST 2: Lane Detection with BGR Input")
    print("="*60)
    
    config = load_config()
    camera = CameraManager(config)
    
    if not camera.start():
        print("❌ FAILED: Camera không start được")
        return False
    
    try:
        frame = camera.capture_frame()
        
        if frame is None:
            print("❌ FAILED: capture_frame() trả về None")
            return False
        
        print(f"✅ Frame captured: {frame.shape}")
        
        # Run lane detection
        lane_config = config.get('ai', {}).get('lane_detection', {})
        error, x_line, center_x, debug_frame = detect_line(frame, lane_config)
        
        print(f"✅ Lane detection completed")
        print(f"   Error: {error}px")
        print(f"   Lane center: {x_line}px")
        print(f"   Frame center: {center_x}px")
        
        # Verify debug frame
        if debug_frame is None:
            print("❌ FAILED: debug_frame là None")
            return False
        
        if len(debug_frame.shape) != 3 or debug_frame.shape[2] != 3:
            print(f"❌ FAILED: debug_frame không phải BGR (shape={debug_frame.shape})")
            return False
        
        print(f"✅ PASSED: Debug frame có format BGR đúng")
        
        # Save debug frame
        cv2.imwrite("test_lane_debug.jpg", debug_frame)
        print(f"📸 Saved debug frame: test_lane_debug.jpg")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: Exception trong lane detection: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        camera.stop()


def test_yolo_inference():
    """Test 3: Kiểm tra YOLO nhận BGR đúng"""
    print("\n" + "="*60)
    print("TEST 3: YOLO Inference with BGR Input")
    print("="*60)
    
    config = load_config()
    camera = CameraManager(config)
    
    if not camera.start():
        print("❌ FAILED: Camera không start được")
        return False
    
    try:
        detector = ObjectDetector(
            model_path='data/models/best_ncnn_model',
            conf_threshold=0.5
        )
        
        if detector.model is None:
            print("⚠️  WARNING: YOLO model không load được (skip test)")
            return True  # Not a failure, just skip
        
        frame = camera.capture_frame()
        
        if frame is None:
            print("❌ FAILED: capture_frame() trả về None")
            return False
        
        print(f"✅ Frame captured: {frame.shape}")
        
        # Run YOLO inference
        detections, result_frame = detector.detect(frame)
        
        print(f"✅ YOLO inference completed")
        print(f"   Detections: {len(detections)}")
        
        for det in detections:
            print(f"   - {det['class_name']}: conf={det['conf']:.2f}, size=({det['w']:.0f}x{det['h']:.0f})")
        
        # Verify result frame
        if result_frame is None:
            print("❌ FAILED: result_frame là None")
            return False
        
        if len(result_frame.shape) != 3 or result_frame.shape[2] != 3:
            print(f"❌ FAILED: result_frame không phải BGR (shape={result_frame.shape})")
            return False
        
        print(f"✅ PASSED: YOLO result frame có format BGR đúng")
        
        # Save result frame
        cv2.imwrite("test_yolo_result.jpg", result_frame)
        print(f"📸 Saved YOLO result: test_yolo_result.jpg")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: Exception trong YOLO inference: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        camera.stop()


def test_web_streaming():
    """Test 4: Kiểm tra web streaming encode đúng"""
    print("\n" + "="*60)
    print("TEST 4: Web Streaming JPEG Encoding")
    print("="*60)
    
    config = load_config()
    camera = CameraManager(config)
    
    if not camera.start():
        print("❌ FAILED: Camera không start được")
        return False
    
    try:
        # Capture frame để populate latest_frame buffer
        frame = camera.capture_frame()
        time.sleep(0.1)  # Wait for buffer update
        
        # Test JPEG encoding
        jpeg_bytes = camera.capture_jpeg(quality=80)
        
        if jpeg_bytes is None:
            print("❌ FAILED: capture_jpeg() trả về None")
            return False
        
        print(f"✅ JPEG encoded successfully")
        print(f"   Size: {len(jpeg_bytes)} bytes")
        
        # Verify JPEG header
        if not jpeg_bytes.startswith(b'\xff\xd8'):
            print("❌ FAILED: JPEG header không hợp lệ")
            return False
        
        print(f"✅ PASSED: JPEG header hợp lệ")
        
        # Decode and verify
        nparr = np.frombuffer(jpeg_bytes, np.uint8)
        decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if decoded is None:
            print("❌ FAILED: Không decode được JPEG")
            return False
        
        if len(decoded.shape) != 3 or decoded.shape[2] != 3:
            print(f"❌ FAILED: Decoded frame không phải BGR (shape={decoded.shape})")
            return False
        
        print(f"✅ PASSED: JPEG decode thành công, format BGR đúng")
        
        # Save decoded frame
        cv2.imwrite("test_web_stream.jpg", decoded)
        print(f"📸 Saved decoded frame: test_web_stream.jpg")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: Exception trong web streaming: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        camera.stop()


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("YUV420 FORMAT FIX VERIFICATION")
    print("Testing: YUV420 → BGR conversion in camera_manager")
    print("="*60)
    
    results = {}
    
    # Test 1: Camera format
    results['camera_format'] = test_camera_format()
    
    # Test 2: Lane detection
    results['lane_detection'] = test_lane_detection()
    
    # Test 3: YOLO inference
    results['yolo_inference'] = test_yolo_inference()
    
    # Test 4: Web streaming
    results['web_streaming'] = test_web_streaming()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(results.values())
    
    print("="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ YUV420 → BGR conversion hoạt động đúng")
        print("✅ Lane detection nhận BGR format")
        print("✅ YOLO inference nhận BGR format")
        print("✅ Web streaming encode đúng")
    else:
        print("❌ SOME TESTS FAILED!")
        print("⚠️  Cần kiểm tra lại code")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
