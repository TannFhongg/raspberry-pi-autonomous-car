"""
Camera Manager - Picamera2 Abstraction Layer
Thread-safe camera access for Raspberry Pi OS Trixie
"""

from picamera2 import Picamera2
import numpy as np
import logging
import threading
import time

logger = logging.getLogger(__name__)


class CameraManager:
    """
    Manages Picamera2 camera with thread-safe access
    Provides easy interface for frame capture and streaming
    
    ✅ FIX: Double buffering để tránh race condition giữa Auto Loop và Web Stream
    """

    def __init__(self, config: dict = None):
        """
        Initialize Camera Manager

        Args:
            config: Hardware configuration dictionary
        """
        self.config = config or {}
        self.camera = None
        self.lock = threading.Lock()
        self.running = False

        # Get camera settings from config
        camera_config = self.config.get("sensors", {}).get("camera", {})
        
        # ============================================================
        # OPTIMIZED: Sử dụng ISP để resize và chuyển format
        # - Xuất thẳng 640x480 từ phần cứng (không dùng CPU resize)
        # - Dùng YUV420 để có sẵn Y channel (Grayscale) miễn phí
        # ============================================================
        self.resolution = tuple(camera_config.get("resolution", [640, 480]))
        self.framerate = camera_config.get("framerate", 30)

        # Picamera2 specific settings
        picam_config = camera_config.get("picamera2", {})
        # YUV420: Y channel = Grayscale, tiết kiệm băng thông và CPU
        self.format = picam_config.get("format", "YUV420")
        self.buffer_count = picam_config.get("buffer_count", 4)

        # ============================================================
        # ✅ FIX RACE CONDITION: Double buffering cho web streaming
        # - latest_frame: Buffer riêng cho web stream (không block auto loop)
        # - latest_frame_lock: Lock riêng cho web stream
        # ============================================================
        self.latest_frame = None
        self.latest_frame_lock = threading.Lock()

        # Performance stats
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0

    def start(self) -> bool:
        """
        Initialize and start camera

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Initializing Picamera2...")

            self.camera = Picamera2()

            # Create video configuration
            video_config = self.camera.create_video_configuration(
                main={"size": self.resolution, "format": self.format},
                buffer_count=self.buffer_count,
            )

            # Configure camera
            self.camera.configure(video_config)

            # Set camera controls if specified
            controls = (
                self.config.get("sensors", {})
                .get("camera", {})
                .get("picamera2", {})
                .get("controls", {})
            )
            if controls:
                # Lọc bỏ bất kỳ giá trị 'None' nào từ config
                # libcamera C++ không chấp nhận 'None', nó mong đợi một số nguyên.
                valid_controls = {
                    key: value for key, value in controls.items() if value is not None
                }

                if valid_controls:
                    logger.info(f"Applying valid camera controls: {valid_controls}")
                    self.camera.set_controls(valid_controls)
                else:
                    # Ghi log nếu tất cả controls đều là 'None' (nhưng không phải là lỗi)
                    logger.info(
                        "No valid camera controls to apply (all were 'None'). Using defaults."
                    )

            # Start camera
            self.camera.start()

            # Wait for camera to stabilize
            time.sleep(0.5)

            # Test capture
            test_frame = self.camera.capture_array()
            if test_frame is None:
                raise Exception("Test capture failed")

            self.running = True
            self.last_fps_time = time.time()

            logger.info(
                f"✓ Picamera2 started: {self.resolution[0]}x{self.resolution[1]} @ {self.framerate}fps"
            )
            logger.info(f"  Format: {self.format}, Buffers: {self.buffer_count}")

            return True

        except Exception as e:
            logger.error(f"✗ Failed to start camera: {e}")
            if self.camera:
                try:
                    self.camera.stop()
                except:
                    pass
                self.camera = None
            return False

    def capture_frame(self) -> np.ndarray:
        """
        Capture a single frame from camera (for Auto Loop - HIGH PRIORITY)
        
        ✅ FIX: Không dùng lock để tránh blocking từ web stream
        ✅ Đồng thời cập nhật buffer cho web stream (non-blocking)

        Returns:
            numpy array in YUV420 format (Y channel = Grayscale), or None if error
        """
        if not self.running or self.camera is None:
            logger.warning("Camera not running")
            return None

        try:
            # ✅ CRITICAL FIX: Capture KHÔNG dùng lock chính
            # → Auto loop không bị block bởi web stream
            frame = self.camera.capture_array()

            # ✅ Cập nhật buffer cho web stream (non-blocking với trylock)
            if self.latest_frame_lock.acquire(blocking=False):
                try:
                    self.latest_frame = frame.copy()  # Copy để tránh reference issue
                finally:
                    self.latest_frame_lock.release()

            # Update FPS counter
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_fps_time >= 1.0:
                self.current_fps = self.frame_count / (
                    current_time - self.last_fps_time
                )
                self.frame_count = 0
                self.last_fps_time = current_time

            return frame

        except Exception as e:
            logger.error(f"Capture error: {e}")
            return None

    def capture_jpeg(self, quality: int = 80) -> bytes:
        """
        Capture frame and encode as JPEG (for Web Stream - LOW PRIORITY)
        
        ✅ FIX: Đọc từ buffer riêng (latest_frame) thay vì gọi capture_frame()
        → Không tranh giành camera với Auto Loop

        Args:
            quality: JPEG quality (1-100)

        Returns:
            JPEG bytes, or None if error
        """
        if not self.running or self.camera is None:
            return None

        try:
            import cv2

            # ✅ CRITICAL FIX: Đọc từ buffer riêng thay vì capture mới
            with self.latest_frame_lock:
                if self.latest_frame is None:
                    return None
                frame = self.latest_frame.copy()

            # YUV420 → BGR để encode JPEG (cho web streaming)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_YUV420p2BGR)
            ret, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ret:
                return buffer.tobytes()
        except Exception as e:
            logger.error(f"JPEG encode error: {e}")

        return None

    def get_fps(self) -> float:
        """Get current FPS"""
        return self.current_fps

    def get_resolution(self) -> tuple:
        """Get camera resolution"""
        return self.resolution

    def is_running(self) -> bool:
        """Check if camera is running"""
        return self.running

    def stop(self):
        """Stop camera"""
        if self.camera:
            try:
                self.camera.stop()
                logger.info("Camera stopped")
            except Exception as e:
                logger.error(f"Error stopping camera: {e}")
            finally:
                self.camera = None

        self.running = False

    def restart(self) -> bool:
        """
        Restart camera

        Returns:
            True if successful
        """
        logger.info("Restarting camera...")
        self.stop()
        time.sleep(0.5)
        return self.start()

    def __del__(self):
        """Destructor"""
        self.stop()

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()


class StreamingCameraManager(CameraManager):
    """
    Extended Camera Manager optimized for video streaming
    Provides MJPEG streaming support for Flask
    """

    def __init__(self, config: dict = None):
        super().__init__(config)
        self.streaming = False
        self.stream_quality = 80

    def start_streaming(self, quality: int = 80):
        """
        Start streaming mode

        Args:
            quality: JPEG quality for streaming
        """
        if not self.running:
            self.start()

        self.stream_quality = quality
        self.streaming = True
        logger.info(f"Streaming started (quality: {quality})")

    def stop_streaming(self):
        """Stop streaming mode"""
        self.streaming = False
        logger.info("Streaming stopped")

    def generate_frames(self):
        """
        Generator for MJPEG streaming

        Yields:
            JPEG frame bytes in multipart format
        """
        self.start_streaming()

        while self.streaming:
            try:
                frame_bytes = self.capture_jpeg(self.stream_quality)

                if frame_bytes:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                    )
                else:
                    # If capture fails, wait and retry
                    time.sleep(0.1)

            except GeneratorExit:
                # Client disconnected
                break
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                break

        self.stop_streaming()


# Singleton instance for web streaming
_web_camera_instance = None
_web_camera_lock = threading.Lock()


def get_web_camera(config: dict = None) -> StreamingCameraManager:
    """
    Get singleton camera instance for web streaming

    Args:
        config: Hardware configuration

    Returns:
        StreamingCameraManager instance
    """
    global _web_camera_instance

    with _web_camera_lock:
        if _web_camera_instance is None:
            _web_camera_instance = StreamingCameraManager(config)

        return _web_camera_instance


def release_web_camera():
    """Release singleton web camera instance"""
    global _web_camera_instance

    with _web_camera_lock:
        if _web_camera_instance:
            _web_camera_instance.stop()
            _web_camera_instance = None
