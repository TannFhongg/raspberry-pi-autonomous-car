"""
Test Camera - Capture ảnh và kiểm tra camera hoạt động
Chạy: python review_tool/test_camera.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
from perception.camera_manager import CameraManager
from utils.config_loader import load_config


def main():
    print("=" * 60)
    print("TEST CAMERA SYSTEM")
    print("=" * 60)
    
    # Load config
    try:
        config = load_config('config/hardware_config.yaml')
        print("✅ Config loaded")
    except Exception as e:
        print(f"❌ Lỗi load config: {e}")
        return
    
    # Initialize camera
    print("\nKhởi tạo camera...")
    camera = CameraManager(config)
    
    if not camera.start():
        print("❌ Không thể khởi động camera!")
        print("\n💡 Kiểm tra:")
        print("  1. Camera có kết nối đúng không?")
        print("  2. Camera có được bật trong raspi-config không?")
        print("  3. Có process nào khác đang dùng camera không?")
        return
    
    print("✅ Camera started")
    
    # Test capture
    print("\n" + "=" * 60)
    print("TEST CAPTURE")
    print("=" * 60)
    
    try:
        # Capture 5 frames để kiểm tra
        print("Capturing 5 frames...")
        for i in range(5):
            frame = camera.capture_frame()
            if frame is None:
                print(f"  Frame {i+1}: ❌ Failed")
            else:
                print(f"  Frame {i+1}: ✅ OK ({frame.shape})")
        
        # Lưu frame cuối
        if frame is not None:
            # Convert YUV420 to BGR if needed
            if len(frame.shape) == 2:  # YUV420 planar
                frame_bgr = cv2.cvtColor(frame[:480, :], cv2.COLOR_YUV420p2BGR)
            else:
                frame_bgr = frame
            
            output_path = "review_tool/test_camera_frame.jpg"
            cv2.imwrite(output_path, frame_bgr)
            print(f"\n✅ Đã lưu frame: {output_path}")
    
    except Exception as e:
        print(f"❌ Lỗi capture: {e}")
    
    # Test FPS
    print("\n" + "=" * 60)
    print("TEST FPS")
    print("=" * 60)
    
    import time
    
    try:
        print("Đo FPS thực tế (3 giây)...")
        frame_count = 0
        start_time = time.time()
        
        while time.time() - start_time < 3.0:
            frame = camera.capture_frame()
            if frame is not None:
                frame_count += 1
        
        elapsed = time.time() - start_time
        fps = frame_count / elapsed
        
        print(f"\n📊 Kết quả:")
        print(f"  Frames captured: {frame_count}")
        print(f"  Time elapsed:    {elapsed:.2f}s")
        print(f"  FPS:             {fps:.1f}")
        
        target_fps = config.get('sensors', {}).get('camera', {}).get('framerate', 30)
        
        if fps >= target_fps * 0.9:
            print(f"  ✅ FPS OK (target: {target_fps})")
        else:
            print(f"  ⚠️  FPS thấp (target: {target_fps})")
    
    except Exception as e:
        print(f"❌ Lỗi test FPS: {e}")
    
    # Cleanup
    camera.stop()
    print("\n✅ Camera stopped")
    
    print("\n" + "=" * 60)
    print("TEST HOÀN TẤT")
    print("=" * 60)


if __name__ == "__main__":
    main()
