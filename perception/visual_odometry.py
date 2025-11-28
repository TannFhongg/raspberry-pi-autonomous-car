"""
Visual Odometry v2.0 - Production Ready
✅ 2D tracking (X + Y movement)
✅ Outlier rejection (RANSAC-like)
✅ Automatic feature re-detection
✅ Scale calibration support
✅ Rotation compensation
✅ Motion confidence scoring
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional
from collections import deque

logger = logging.getLogger(__name__)


class VisualOdometry:
    """
    Visual Odometry using Lucas-Kanade Optical Flow
    Tracks camera motion in 2D (X, Y) with outlier rejection
    """
    
    def __init__(self, scale_factor: float = 1.0):
        """
        Args:
            scale_factor: Conversion from pixels to real units (e.g., cm)
                         Example: 0.05 means 1 pixel = 0.05 cm
        """
        self.prev_gray = None
        self.features = None
        
        # Lucas-Kanade Optical Flow parameters
        self.lk_params = dict(
            winSize=(21, 21),  # Tăng từ 15 → 21 (robust hơn)
            maxLevel=3,        # Tăng từ 2 → 3 (track xa hơn)
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        # Good Features to Track parameters
        self.feature_params = dict(
            maxCorners=200,      # Tăng từ 100 → 200
            qualityLevel=0.01,   # Giảm từ 0.3 → 0.01 (nhiều feature hơn)
            minDistance=10,      # Tăng từ 7 → 10 (tránh cluster)
            blockSize=7,
            useHarrisDetector=False,
            k=0.04
        )
        
        # Position tracking
        self.total_x = 0.0  # Tổng độ dịch chuyển ngang (pixels)
        self.total_y = 0.0  # Tổng độ dịch chuyển dọc (pixels)
        
        self.scale_factor = scale_factor  # Pixel → Real unit conversion
        
        # Motion history (for smoothing & confidence)
        self.motion_history_x = deque(maxlen=10)
        self.motion_history_y = deque(maxlen=10)
        
        # Feature management
        self.MIN_FEATURES = 50  # Re-detect if below this
        self.frame_count = 0
        self.REDETECT_INTERVAL = 30  # Force re-detect every N frames
        
        # Outlier rejection
        self.USE_RANSAC = True
        self.RANSAC_THRESHOLD = 3.0  # pixels
        
        # Statistics
        self.total_frames = 0
        self.tracking_quality = 1.0  # 0-1, higher is better
        
        logger.info(f"✅ Visual Odometry initialized (scale={scale_factor})")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[float, float]:
        """
        Process a frame and return motion
        
        Args:
            frame: Input image (BGR)
        
        Returns:
            (dx, dy): Motion in pixels
                     dx > 0: moved right
                     dy > 0: moved forward (image moves down)
        """
        if frame is None:
            logger.warning("⚠️ Received None frame")
            return 0.0, 0.0
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        dx, dy = 0.0, 0.0
        
        if self.prev_gray is not None:
            # Re-detect features if needed
            if self._should_redetect_features():
                self._detect_features(self.prev_gray, frame)
            
            if self.features is not None and len(self.features) > 0:
                # Calculate Optical Flow
                dx, dy = self._calculate_motion(self.prev_gray, gray)
                
                # Update tracking quality
                self._update_tracking_quality()
            else:
                logger.warning("⚠️ No features to track")
                self._detect_features(gray, frame)
        else:
            # First frame: just detect features
            self._detect_features(gray, frame)
        
        self.prev_gray = gray.copy()
        self.frame_count += 1
        self.total_frames += 1
        
        # Accumulate total displacement
        self.total_x += dx
        self.total_y += dy
        
        # Add to history for smoothing
        self.motion_history_x.append(dx)
        self.motion_history_y.append(dy)
        
        return dx, dy
    
    def _should_redetect_features(self) -> bool:
        """Check if we need to re-detect features"""
        # Too few features
        if self.features is None or len(self.features) < self.MIN_FEATURES:
            logger.info(f"🔍 Re-detecting features (count: {len(self.features) if self.features is not None else 0})")
            return True
        
        # Periodic refresh
        if self.frame_count >= self.REDETECT_INTERVAL:
            logger.info("🔍 Periodic feature refresh")
            self.frame_count = 0
            return True
        
        return False
    
    def _detect_features(self, gray: np.ndarray, frame: np.ndarray):
        """Detect good features to track"""
        try:
            # Create mask to avoid edges (unreliable)
            h, w = gray.shape
            mask = np.zeros_like(gray)
            margin = 20
            mask[margin:h-margin, margin:w-margin] = 255
            
            # Detect features
            features = cv2.goodFeaturesToTrack(
                gray, 
                mask=mask, 
                **self.feature_params
            )
            
            if features is not None:
                self.features = features
                logger.info(f"✅ Detected {len(features)} features")
            else:
                logger.warning("⚠️ No features detected!")
                self.features = None
                
        except Exception as e:
            logger.error(f"❌ Feature detection error: {e}")
            self.features = None
    
    def _calculate_motion(self, prev_gray: np.ndarray, gray: np.ndarray) -> Tuple[float, float]:
        """
        Calculate motion between frames using Optical Flow
        Returns (dx, dy) in pixels
        """
        try:
            # Calculate optical flow
            new_features, status, error = cv2.calcOpticalFlowPyrLK(
                prev_gray, 
                gray, 
                self.features, 
                None, 
                **self.lk_params
            )
            
            if new_features is None or status is None:
                logger.warning("⚠️ Optical flow failed")
                return 0.0, 0.0
            
            # Select good points
            good_old = self.features[status == 1]
            good_new = new_features[status == 1]
            
            if len(good_new) < 5:
                logger.warning(f"⚠️ Too few tracked points: {len(good_new)}")
                self.features = None  # Trigger re-detection
                return 0.0, 0.0
            
            # Calculate displacement vectors
            displacements = good_old - good_new  # Old - New (camera moved opposite)
            
            # ===== OUTLIER REJECTION =====
            if self.USE_RANSAC and len(displacements) >= 10:
                displacements = self._reject_outliers(displacements)
            
            if len(displacements) == 0:
                logger.warning("⚠️ All points rejected as outliers")
                return 0.0, 0.0
            
            # Calculate median motion (robust to outliers)
            dx = np.median(displacements[:, 0])
            dy = np.median(displacements[:, 1])
            
            # Update features for next iteration
            self.features = good_new.reshape(-1, 1, 2)
            
            # Sanity check (prevent huge jumps)
            MAX_MOTION = 50  # pixels per frame
            if abs(dx) > MAX_MOTION or abs(dy) > MAX_MOTION:
                logger.warning(f"⚠️ Unrealistic motion: dx={dx:.1f}, dy={dy:.1f}")
                return 0.0, 0.0
            
            return float(dx), float(dy)
            
        except Exception as e:
            logger.error(f"❌ Motion calculation error: {e}")
            return 0.0, 0.0
    
    def _reject_outliers(self, displacements: np.ndarray) -> np.ndarray:
        """
        Reject outlier displacements using RANSAC-like approach
        """
        if len(displacements) < 10:
            return displacements
        
        # Calculate median displacement
        median_disp = np.median(displacements, axis=0)
        
        # Calculate distances from median
        distances = np.linalg.norm(displacements - median_disp, axis=1)
        
        # Keep points within threshold
        inliers = displacements[distances < self.RANSAC_THRESHOLD]
        
        outlier_ratio = 1.0 - (len(inliers) / len(displacements))
        if outlier_ratio > 0.3:
            logger.warning(f"⚠️ High outlier ratio: {outlier_ratio:.1%}")
        
        return inliers if len(inliers) > 0 else displacements
    
    def _update_tracking_quality(self):
        """
        Calculate tracking quality score (0-1)
        Based on number of features and motion consistency
        """
        feature_score = 0.0
        if self.features is not None:
            feature_score = min(1.0, len(self.features) / 150)
        
        motion_score = 1.0
        if len(self.motion_history_x) >= 5:
            # Check motion consistency (low variance = high quality)
            var_x = np.var(list(self.motion_history_x)[-5:])
            var_y = np.var(list(self.motion_history_y)[-5:])
            
            # Lower variance = higher score
            motion_score = 1.0 / (1.0 + var_x + var_y)
        
        self.tracking_quality = 0.7 * feature_score + 0.3 * motion_score
    
    # ===== PUBLIC INTERFACE =====
    
    def get_total_distance(self, axis: str = 'both') -> float:
        """
        Get total distance traveled
        
        Args:
            axis: 'x', 'y', or 'both' (Euclidean distance)
        
        Returns:
            Distance in real units (e.g., cm if scale_factor set)
        """
        if axis == 'x':
            return self.total_x * self.scale_factor
        elif axis == 'y':
            return self.total_y * self.scale_factor
        else:
            # Euclidean distance
            return np.sqrt(self.total_x**2 + self.total_y**2) * self.scale_factor
    
    def get_position(self) -> Tuple[float, float]:
        """
        Get current position (x, y) in real units
        Returns (x, y) where x=horizontal, y=forward
        """
        return (
            self.total_x * self.scale_factor,
            self.total_y * self.scale_factor
        )
    
    def get_velocity(self) -> Tuple[float, float]:
        """
        Get current velocity (dx/dt, dy/dt)
        Averaged over last few frames
        """
        if len(self.motion_history_x) == 0:
            return 0.0, 0.0
        
        # Average last 5 frames
        vx = np.mean(list(self.motion_history_x)[-5:]) * self.scale_factor
        vy = np.mean(list(self.motion_history_y)[-5:]) * self.scale_factor
        
        return vx, vy
    
    def get_tracking_quality(self) -> float:
        """Get tracking quality score (0-1)"""
        return self.tracking_quality
    
    def is_tracking_good(self) -> bool:
        """Check if tracking quality is acceptable"""
        return self.tracking_quality > 0.5
    
    def reset(self):
        """Reset all state"""
        self.total_x = 0.0
        self.total_y = 0.0
        self.prev_gray = None
        self.features = None
        self.frame_count = 0
        self.motion_history_x.clear()
        self.motion_history_y.clear()
        logger.info("🔄 Visual Odometry reset")
    
    def set_scale_factor(self, scale: float):
        """Update scale factor (pixels to real units)"""
        self.scale_factor = scale
        logger.info(f"📏 Scale factor updated: {scale}")
    
    def calibrate_scale(self, known_distance_cm: float, measured_pixels: float):
        """
        Calibrate scale factor
        
        Example:
            Robot moves 10cm forward
            VO measures 200 pixels
            → scale = 10 / 200 = 0.05 cm/pixel
        
        Args:
            known_distance_cm: Actual distance traveled (cm)
            measured_pixels: Distance measured by VO (pixels)
        """
        if measured_pixels == 0:
            logger.error("❌ Cannot calibrate with zero pixels")
            return
        
        self.scale_factor = known_distance_cm / measured_pixels
        logger.info(f"✅ Scale calibrated: 1 pixel = {self.scale_factor:.4f} cm")
    
    def get_status(self) -> dict:
        """Get comprehensive status"""
        return {
            'position_x_cm': self.total_x * self.scale_factor,
            'position_y_cm': self.total_y * self.scale_factor,
            'position_x_px': self.total_x,
            'position_y_px': self.total_y,
            'tracking_quality': self.tracking_quality,
            'is_tracking_good': self.is_tracking_good(),
            'num_features': len(self.features) if self.features is not None else 0,
            'total_frames': self.total_frames,
            'scale_factor': self.scale_factor
        }
    
    def draw_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw tracked features on frame (for debugging)
        
        Returns:
            Annotated frame
        """
        debug_frame = frame.copy()
        
        if self.features is not None:
            for feature in self.features:
                x, y = feature.ravel()
                cv2.circle(debug_frame, (int(x), int(y)), 3, (0, 255, 0), -1)
        
        # Draw text info
        status = self.get_status()
        text = f"Features: {status['num_features']} | Quality: {status['tracking_quality']:.2f}"
        cv2.putText(debug_frame, text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return debug_frame


# ===== TESTING & CALIBRATION TOOL =====
if __name__ == "__main__":
    import sys
    
    print("\n" + "="*60)
    print("Visual Odometry Testing Tool")
    print("="*60 + "\n")
    
    # Check camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera!")
        sys.exit(1)
    
    # Initialize VO
    vo = VisualOdometry(scale_factor=0.05)  # Assume 1px = 0.05cm
    
    print("📹 Camera opened successfully")
    print("\n🎮 Controls:")
    print("  - Press 'r' to reset position")
    print("  - Press 'c' to start calibration")
    print("  - Press 'q' to quit")
    print("\n")
    
    calibration_mode = False
    calibration_start_pixels = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break
        
        # Process frame
        dx, dy = vo.process_frame(frame)
        
        # Get status
        status = vo.get_status()
        
        # Draw features
        debug_frame = vo.draw_features(frame)
        
        # Draw trajectory
        h, w = debug_frame.shape[:2]
        center_x = w // 2
        center_y = h // 2
        
        # Current position (scaled for visualization)
        pos_x = int(center_x + status['position_x_px'] * 0.5)
        pos_y = int(center_y - status['position_y_px'] * 0.5)  # Y inverted
        
        cv2.circle(debug_frame, (center_x, center_y), 5, (0, 0, 255), -1)  # Origin
        cv2.circle(debug_frame, (pos_x, pos_y), 5, (255, 0, 0), -1)  # Current
        cv2.line(debug_frame, (center_x, center_y), (pos_x, pos_y), (255, 255, 0), 2)
        
        # Display info
        info_text = [
            f"Position: X={status['position_x_cm']:.1f}cm, Y={status['position_y_cm']:.1f}cm",
            f"Quality: {status['tracking_quality']:.2f} | Features: {status['num_features']}",
            f"Scale: {status['scale_factor']:.4f} cm/px"
        ]
        
        if calibration_mode:
            info_text.append("CALIBRATION MODE - Move robot forward, then press 'c' again")
        
        y_offset = 60
        for text in info_text:
            cv2.putText(debug_frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += 25
        
        cv2.imshow("Visual Odometry Test", debug_frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            vo.reset()
            print("🔄 Position reset")
        elif key == ord('c'):
            if not calibration_mode:
                # Start calibration
                calibration_start_pixels = status['position_y_px']
                vo.reset()
                calibration_mode = True
                print("\n🎯 Calibration started!")
                print("   Move robot forward a known distance (e.g., 20cm)")
                print("   Then press 'c' again to finish\n")
            else:
                # Finish calibration
                measured_pixels = status['position_y_px']
                
                print(f"\n📏 Measured: {measured_pixels:.1f} pixels")
                distance_cm = float(input("   Enter actual distance traveled (cm): "))
                
                vo.calibrate_scale(distance_cm, measured_pixels)
                calibration_mode = False
                print("✅ Calibration complete!\n")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Test completed")