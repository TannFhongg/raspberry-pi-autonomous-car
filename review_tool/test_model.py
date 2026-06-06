"""
Test NCNN Model với 1 ảnh tĩnh - KHÔNG cần camera
Chạy: python review_tool/test_model.py --image test_full_hd.jpg
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from utils.config_loader import load_config
from perception.object_detector import ObjectDetector


def main():
    parser = argparse.ArgumentParser(description="Test NCNN model với ảnh tĩnh")
    parser.add_argument("--image", default="test_full_hd.jpg", help="Đường dẫn đến ảnh test")
    parser.add_argument("--config", default="config/hardware_config.yaml", help="File cấu hình")
    args = parser.parse_args()
    
    # Load config
    print("=" * 60)
    print("TEST NCNN MODEL - OBJECT DETECTION")
    print("=" * 60)
    
    try:
        cfg = load_config(args.config)
        print(f"✅ Loaded config: {args.config}")
    except Exception as e:
        print(f"❌ Không thể load config: {e}")
        return
    
    # Kiểm tra file ảnh
    if not Path(args.image).exists():
        print(f"\n❌ Ảnh không tồn tại: {args.image}")
        print("💡 Chạy 'python capture.py' để chụp ảnh mẫu trước")
        return
    
    # Load ảnh
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"❌ Không thể đọc ảnh: {args.image}")
        return
    
    print(f"✅ Đã load ảnh: {args.image} ({frame.shape[1]}×{frame.shape[0]})")
    
    # Initialize Object Detector
    print("\n" + "=" * 60)
    print("Khởi tạo Object Detector...")
    print("=" * 60)
    
    try:
        detector = ObjectDetector(
            model_path='models/best_ncnn_model',
            conf_threshold=0.5
        )
        print("✅ Object Detector initialized")
    except Exception as e:
        print(f"❌ Lỗi khởi tạo detector: {e}")
        return
    
    # Inference test
    print("\n" + "=" * 60)
    print("INFERENCE TEST")
    print("=" * 60)
    
    # Warm-up (bỏ qua kết quả)
    print("Warm-up model...")
    detector.detect(frame)
    
    # Đo thời gian inference (10 lần)
    print("Đo inference time (10 lần)...")
    times = []
    
    for i in range(10):
        t0 = time.monotonic()
        detections, _ = detector.detect(frame)
        elapsed = (time.monotonic() - t0) * 1000
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.1f}ms")
    
    avg_ms = sum(times) / len(times)
    max_fps = 1000 / avg_ms
    
    print(f"\n📊 Kết quả:")
    print(f"  Inference avg: {avg_ms:.1f}ms")
    print(f"  FPS tối đa:    {max_fps:.1f}")
    
    # Đánh giá performance
    if avg_ms < 100:
        print("  ✅ Model speed: EXCELLENT (< 100ms)")
    elif avg_ms < 200:
        print("  ⚠️  Model speed: OK (< 200ms)")
    else:
        print(f"  ❌ Model speed: SLOW ({avg_ms:.0f}ms)")
        print("  💡 Khuyến nghị: Giảm input_size hoặc dùng model nhẹ hơn")
    
    # Detection test
    print("\n" + "=" * 60)
    print("DETECTION TEST")
    print("=" * 60)
    
    detections, result_frame = detector.detect(frame)
    
    print(f"Phát hiện {len(detections)} đối tượng:")
    
    if detections:
        for i, det in enumerate(detections):
            print(f"\n  [{i+1}] {det['class_name']}")
            print(f"      Confidence: {det['conf']:.2%}")
            print(f"      Position:   ({det['x']:.0f}, {det['y']:.0f})")
            print(f"      Size:       {det['w']:.0f}×{det['h']:.0f}px")
        
        # Vẽ bounding boxes
        for det in detections:
            x, y, w, h = int(det['x']), int(det['y']), int(det['w']), int(det['h'])
            x1 = int(x - w/2)
            y1 = int(y - h/2)
            x2 = int(x + w/2)
            y2 = int(y + h/2)
            
            # Vẽ box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Vẽ label
            label = f"{det['class_name']} {det['conf']:.0%}"
            cv2.putText(result_frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Lưu kết quả
        output_path = "review_tool/test_model_result.jpg"
        cv2.imwrite(output_path, result_frame)
        print(f"\n✅ Đã lưu kết quả: {output_path}")
    else:
        print("  ⚠️  Không phát hiện đối tượng nào")
    
    print("\n" + "=" * 60)
    print("TEST HOÀN TẤT")
    print("=" * 60)


if __name__ == "__main__":
    main()
