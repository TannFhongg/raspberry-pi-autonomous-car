#!/usr/bin/env python3
"""
Visual Odometry Calibration Tool
Giúp tính scale factor chính xác
"""

import cv2
import sys
sys.path.append('.')

from perception.visual_odometry import VisualOdometry

def main():
    print("\n" + "="*60)
    print("Visual Odometry Calibration Tool")
    print("="*60 + "\n")
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera!")
        return
    
    vo = VisualOdometry(scale_factor=1.0)  # Start with 1.0
    
    print("📹 Camera opened")
    print("\n🎮 Instructions:")
    print("  1. Press 'c' to START calibration")
    print("  2. Move robot STRAIGHT for a known distance (e.g., 20cm)")
    print("  3. Press 'c' AGAIN when done")
    print("  4. Enter actual distance traveled")
    print("  5. Press 'q' to quit\n")
    
    calibration_mode = False
    start_y = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process VO
        dx, dy = vo.process_frame(frame)
        
        # Draw features
        debug_frame = vo.draw_features(frame)
        
        # Show status
        status = vo.get_status()
        info = [
            f"Y Position: {status['position_y_px']:.1f} px ({status['position_y_cm']:.1f} cm)",
            f"Quality: {status['tracking_quality']:.2f}",
            f"Features: {status['num_features']}"
        ]
        
        if calibration_mode:
            info.append("⏺️ CALIBRATION MODE - Move robot now!")
        
        y = 30
        for text in info:
            cv2.putText(debug_frame, text, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y += 30
        
        cv2.imshow("VO Calibration", debug_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('c'):
            if not calibration_mode:
                # Start
                start_y = status['position_y_px']
                vo.reset()
                calibration_mode = True
                print("\n🎯 Calibration STARTED!")
                print("   Move robot forward now...")
            else:
                # Finish
                measured_pixels = status['position_y_px']
                print(f"\n📏 Measured: {measured_pixels:.1f} pixels")
                
                distance_cm = float(input("   Enter actual distance (cm): "))
                
                vo.calibrate_scale(distance_cm, measured_pixels)
                
                print(f"\n✅ Calibration Complete!")
                print(f"   Scale Factor: {vo.scale_factor:.4f} cm/pixel")
                print(f"\n💡 Add this to config.yaml:")
                print(f"   visual_odometry:")
                print(f"     scale_factor: {vo.scale_factor:.4f}\n")
                
                calibration_mode = False
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()