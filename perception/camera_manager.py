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
        self.resolution = tuple(camera_config.get("resolution", [1640, 1232]))
        self.framerate = camera_config.get("framerate", 30)

        # Picamera2 specific settings
        picam_config = camera_config.get("picamera2", {})
        self.format = picam_config.get("format", "RGB888")
        self.buffer_count = picam_config.get("buffer_count", 4)

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
        Capture a single frame from camera

        Returns:
            numpy array in RGB format, or None if error
        """
        if not self.running or self.camera is None:
            logger.warning("Camera not running")
            return None

        try:
            with self.lock:
                frame = self.camera.capture_array()

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
        Capture frame and encode as JPEG

        Args:
            quality: JPEG quality (1-100)

        Returns:
            JPEG bytes, or None if error
        """
        frame = self.capture_frame()
        if frame is None:
            return None

        try:
            import cv2

            # Frame đã là BGR, nén thẳng luôn
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
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
