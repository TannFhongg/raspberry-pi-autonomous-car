"""
Test Lane Detection với ảnh tĩnh
Chạy: python review_tool/test_lane_detection.py --image test_full_hd.jpg
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
from perception.lane_detector import detect_line, calibrate_lane_width
from utils.config_loader import load_config


def main():
    parser = argparse.ArgumentParser(description="Test lane detection")
    parser.add_argument("--image", default="test_full_hd.jpg", help="Đường dẫn ảnh test")
    parser.add_argument("--calibrate", action="store_true", help="Chạy calibration tool")
    parser.add_argument("--config", default="config/hardware_config.yaml", help="File cấu hình")
    args = parser.parse_args()
    
    print("=" * 60)
    print("TEST LANE DETECTION")
    print("=" * 60)
    
    # Load config
    try:
        cfg = load_config(args.config)
        lane_config = cfg.get('ai', {}).get('lane_detection', {})
        print(f"✅ Loaded config: {args.config}")
    except Exception as e:
        print(f"❌ Không thể load config: {e}")
        lane_config = None
    
    # Kiểm tra file ảnh
    if not Path(args.image).exists():
        print(f"\n❌ Ảnh không tồn tại: {args.image}")
        print("💡 Chạy 'python review_tool/test_camera.py' để chụp ảnh mẫu trước")
        return
    
    # Load ảnh
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"❌ Không thể đọc ảnh: {args.image}")
        return
    
    print(f"✅ Đã load ảnh: {args.image} ({frame.shape[1]}×{frame.shape[0]})")
    
    # Calibration mode
    if args.calibrate:
        print("\n" + "=" * 60)
        print("CALIBRATION MODE")
        print("=" * 60)
        print("Đo lane width (25cm thật → pixels)...")
        
        lane_width = calibrate_lane_width(frame, show_result=True)
        
        if lane_width:
            print(f"\n✅ CALIBRATION RESULT:")
            print(f"  Lane width: {lane_width} pixels")
            print(f"\n💡 Cập nhật vào config/hardware_config.yaml:")
            print(f"  ai.lane_detection.lane_width_pixels: {lane_width}")
        
        return
    
    # Lane detection test
    print("\n" + "=" * 60)
    print("LANE DETECTION TEST")
    print("=" * 60)
    
    try:
        error, x_line, center_x, debug_frame = detect_line(frame, lane_config, debug=True)
        
        print(f"\n📊 Kết quả:")
        print(f"  Error:      {error:+4d} px")
        print(f"  Line X:     {x_line} px")
        print(f"  Center X:   {center_x} px")
        
        # Đánh giá
        if abs(error) <= 10:
            print(f"  ✅ Lane centered (error < 10px)")
        elif abs(error) <= 50:
            print(f"  ⚠️  Slight deviation (10-50px)")
        elif abs(error) <= 110:
            print(f"  ⚠️  Large deviation (50-110px)")
        else:
            print(f"  ❌ Lane lost (error > 110px)")
        
        # Lưu debug frame
        if debug_frame is not None:
            output_path = "review_tool/test_lane_result.jpg"
            cv2.imwrite(output_path, debug_frame)
            print(f"\n✅ Đã lưu debug frame: {output_path}")
    
    except Exception as e:
        print(f"❌ Lỗi lane detection: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("TEST HOÀN TẤT")
    print("=" * 60)


if __name__ == "__main__":
    main()
