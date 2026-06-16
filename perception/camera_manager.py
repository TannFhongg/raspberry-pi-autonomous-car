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


def _parse_size(value, default=None, name="size"):
    """Parse a [width, height] style config value."""
    if value is None:
        return default

    try:
        width, height = value
        width = int(width)
        height = int(height)
    except (TypeError, ValueError):
        logger.warning("Invalid camera %s=%r; using %r", name, value, default)
        return default

    if width <= 0 or height <= 0:
        logger.warning("Invalid camera %s=%r; using %r", name, value, default)
        return default

    return (width, height)


def yuv420_to_bgr(frame_yuv: np.ndarray) -> np.ndarray:
    """
    Convert Picamera2 YUV420 planar frames to BGR.

    Picamera2 returns I420 layout (Y, U, V). OpenCV's generic
    COLOR_YUV420p2BGR is an alias for YV12, so it swaps U/V and shifts colors.
    """
    import cv2

    return cv2.cvtColor(frame_yuv, cv2.COLOR_YUV2BGR_I420)


def frame_to_bgr(frame: np.ndarray, format_name: str = None) -> np.ndarray:
    """Convert a captured Picamera2 frame to OpenCV BGR."""
    if frame is None:
        return None

    import cv2

    format_name = str(format_name or "").upper()

    if format_name == "YUV420":
        return yuv420_to_bgr(frame)

    if len(frame.shape) == 2:
        if _looks_like_yuv420_frame(frame):
            return yuv420_to_bgr(frame)
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    if len(frame.shape) == 3:
        channels = frame.shape[2]
        if channels == 3:
            # Picamera2/libcamera's RGB888 capture array on this platform is
            # already in the channel order expected by OpenCV consumers here.
            # Swapping it again turns yellow objects blue/cyan.
            if format_name == "RGB888":
                return frame
            if format_name.startswith("RGB"):
                return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame
        if channels == 4:
            if format_name.startswith("RGB"):
                return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    return frame


def _looks_like_yuv420_frame(frame: np.ndarray) -> bool:
    if frame is None or len(frame.shape) != 2:
        return False

    height, width = frame.shape[:2]
    image_height = (height * 2) // 3
    return height % 3 == 0 and image_height > 0 and image_height % 2 == 0 and width % 2 == 0


def crop_yuv420_frame(frame_yuv: np.ndarray, resolution: tuple) -> np.ndarray:
    """Crop Picamera2 YUV420 stride padding to the configured image size."""
    if frame_yuv is None or len(frame_yuv.shape) != 2:
        return frame_yuv

    width, height = resolution
    expected_shape = ((height * 3) // 2, width)
    if frame_yuv.shape[0] < expected_shape[0] or frame_yuv.shape[1] < expected_shape[1]:
        return frame_yuv

    if frame_yuv.shape == expected_shape and frame_yuv.flags["C_CONTIGUOUS"]:
        return frame_yuv

    return np.ascontiguousarray(frame_yuv[: expected_shape[0], : expected_shape[1]])


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
        self.capture_lock = threading.Lock()
        self.running = False

        # Get camera settings from config
        camera_config = self.config.get("sensors", {}).get("camera", {})
        
        # Use the ISP to resize before frames reach Python.
        self.resolution = _parse_size(
            camera_config.get("resolution"), (960, 720), "resolution"
        )
        self.framerate = camera_config.get("framerate", 30)

        picam_config = camera_config.get("picamera2", {})
        self.format = picam_config.get("format", "YUV420")
        self.buffer_count = picam_config.get("buffer_count", 4)
        self.sensor_output_size = _parse_size(
            picam_config.get(
                "sensor_output_size", camera_config.get("sensor_output_size")
            ),
            None,
            "sensor_output_size",
        )
        self.sensor_bit_depth = int(
            picam_config.get("sensor_bit_depth", camera_config.get("sensor_bit_depth", 10))
        )
        self.sensor_config = {}
        if self.sensor_output_size:
            self.sensor_config = {
                "output_size": self.sensor_output_size,
                "bit_depth": self.sensor_bit_depth,
            }

        # ============================================================
        # ✅ FIX RACE CONDITION: Double buffering cho web streaming
        # - latest_frame: Buffer riêng cho web stream (không block auto loop)
        # - latest_frame_lock: Lock riêng cho web stream
        # ============================================================
        self.latest_frame = None
        self.latest_frame_time = 0.0
        self.latest_frame_lock = threading.Lock()
        self.frame_stale_after = max(1.0 / max(float(self.framerate), 1.0), 0.01)

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
        with self.lock:
            if self.running and self.camera is not None:
                return True

            try:
                logger.info("Initializing Picamera2...")

                self.camera = Picamera2()

                # Create video configuration
                config_kwargs = {
                    "main": {"size": self.resolution, "format": self.format},
                    "buffer_count": self.buffer_count,
                }
                if self.sensor_config:
                    config_kwargs["sensor"] = self.sensor_config

                video_config = self.camera.create_video_configuration(**config_kwargs)

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

                # Test capture and populate the web stream buffer immediately.
                with self.capture_lock:
                    test_frame = self.camera.capture_array()
                if test_frame is None:
                    raise Exception("Test capture failed")
                test_frame = self._normalize_captured_frame(test_frame)
                self._store_latest_frame(test_frame)

                self.running = True
                self.last_fps_time = time.time()

                logger.info(
                    f"✓ Picamera2 started: {self.resolution[0]}x{self.resolution[1]} @ {self.framerate}fps"
                )
                logger.info(f"  Format: {self.format}, Buffers: {self.buffer_count}")
                if self.sensor_config:
                    logger.info(f"  Sensor: {self.sensor_config}")

                return True

            except Exception as e:
                logger.error(f"✗ Failed to start camera: {e}")
                if self.camera:
                    try:
                        self.camera.stop()
                    except Exception as stop_error:
                        logger.error(f"Error cleaning up failed camera start: {stop_error}")
                    self.camera = None
                self.running = False
                return False

    def _store_latest_frame(self, frame: np.ndarray):
        """Copy a captured frame into the stream buffer."""
        with self.latest_frame_lock:
            self.latest_frame = frame.copy()
            self.latest_frame_time = time.time()

    def _get_latest_frame_copy(self):
        """Return a copy of the latest frame and its timestamp."""
        with self.latest_frame_lock:
            if self.latest_frame is None:
                return None, 0.0
            return self.latest_frame.copy(), self.latest_frame_time

    def _update_fps(self):
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.frame_count / (
                current_time - self.last_fps_time
            )
            self.frame_count = 0
            self.last_fps_time = current_time

    def _normalize_captured_frame(self, frame: np.ndarray) -> np.ndarray:
        if str(self.format).upper() == "YUV420":
            return crop_yuv420_frame(frame, self.resolution)
        return frame

    def _capture_raw_frame(self) -> np.ndarray:
        """Capture one raw frame from Picamera2 with camera access serialized."""
        if not self.running or self.camera is None:
            logger.warning("Camera capture requested while camera is not running")
            return None

        with self.capture_lock:
            if not self.running or self.camera is None:
                logger.warning("Camera stopped before capture could start")
                return None
            frame = self.camera.capture_array()

        if frame is None:
            logger.error("Camera capture returned None")
            return None

        frame = self._normalize_captured_frame(frame)
        self._store_latest_frame(frame)
        self._update_fps()
        return frame

    def capture_frame(self) -> np.ndarray:
        """
        Capture a single frame from camera (for Auto Loop - HIGH PRIORITY)
        
        ✅ Camera capture được serialize để tránh Picamera2 race giữa các thread
        ✅ Đồng thời cập nhật buffer cho web stream (ALWAYS update, no skip)
        ✅ OPTIMIZED: Trả về RAW YUV420 để tận dụng ISP hardware

        Returns:
            numpy array in YUV420 format (raw from ISP), or None if error
        """
        if not self.running or self.camera is None:
            logger.warning("Camera not running")
            return None

        try:
            frame_yuv = self._capture_raw_frame()
            if frame_yuv is None:
                return None
            
            # ============================================================
            # ✅ OPTIMIZED: TRẢ VỀ RAW YUV420 - KHÔNG CONVERT
            # ============================================================
            # Picamera2 với format='YUV420' trả về shape (H*3//2, W) - 2D array
            # Layout: Y plane (H×W) + U plane (H/2×W/2) + V plane (H/2×W/2)
            # 
            # KIẾN TRÚC CHUẨN:
            # 1. Lane detector: Lấy trực tiếp kênh Y (grayscale) - CPU cost = 0
            # 2. YOLO detector: Convert YUV420→BGR chỉ khi cần - CPU cost minimal
            # 3. Tận dụng tối đa ISP hardware (Y channel = grayscale miễn phí)
            # ============================================================
            
            frame = frame_yuv  # Trả về raw YUV420

            return frame

        except Exception as e:
            logger.error(f"Capture error: {e}")
            return None

    def capture_jpeg(self, quality: int = 80) -> bytes:
        """
        Capture frame and encode as JPEG (for Web Stream - LOW PRIORITY)
        
        ✅ FIX: Đọc từ buffer riêng (latest_frame) thay vì gọi capture_frame()
        → Không tranh giành camera với Auto Loop
        ✅ OPTIMIZED: Convert YUV420→BGR chỉ khi cần (cho web stream)
        ✅ FIX STALE FRAME: Copy frame TRONG lock để tránh race condition

        Args:
            quality: JPEG quality (1-100)

        Returns:
            JPEG bytes, or None if error
        """
        try:
            import cv2

            try:
                quality = int(quality)
            except (TypeError, ValueError):
                logger.warning(f"Invalid JPEG quality {quality!r}; using 80")
                quality = 80
            quality = max(1, min(100, quality))

            if not self.running or self.camera is None:
                logger.warning("JPEG capture requested while camera is not running")
                return None

            # ============================================================
            # ✅ FIX STALE FRAME: Copy frame TRONG lock (atomic read)
            # ============================================================
            # Lock chỉ để copy pointer/data (< 1ms cho 640x480x3)
            # Encode JPEG NGOÀI lock để không block capture_frame()
            # 
            # Timeline:
            # 1. Lock → Copy frame (< 1ms) → Unlock
            # 2. Convert YUV420→BGR (nếu cần) + Encode JPEG NGOÀI lock
            # ============================================================
            frame_yuv, frame_time = self._get_latest_frame_copy()
            if frame_yuv is None or (time.time() - frame_time) >= self.frame_stale_after:
                fresh_frame = self._capture_raw_frame()
                if fresh_frame is not None:
                    frame_yuv = fresh_frame.copy()

            if frame_yuv is None:
                logger.warning("No camera frame available for JPEG encoding")
                return None

            frame_bgr = frame_to_bgr(frame_yuv, self.format)

            # Encode JPEG NGOÀI lock (5-10ms, không block capture_frame)
            ret, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ret:
                return buffer.tobytes()
            logger.error("JPEG encode failed: cv2.imencode returned False")
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
        with self.lock:
            self.running = False
            if self.camera:
                with self.capture_lock:
                    try:
                        self.camera.stop()
                        logger.info("Camera stopped")
                    except Exception as e:
                        logger.error(f"Error stopping camera: {e}")
                    finally:
                        self.camera = None

            with self.latest_frame_lock:
                self.latest_frame = None
                self.latest_frame_time = 0.0

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
        self.stream_lock = threading.Lock()
        self.active_stream_clients = 0
        self.stream_frame_delay = self.frame_stale_after

    def start_streaming(self, quality: int = 80):
        """
        Start streaming mode

        Args:
            quality: JPEG quality for streaming
        """
        try:
            quality = int(quality)
        except (TypeError, ValueError):
            logger.warning(f"Invalid stream quality {quality!r}; using 80")
            quality = 80
        quality = max(1, min(100, quality))

        if not self.running and not self.start():
            logger.error("Streaming start failed: camera could not start")
            return False

        with self.stream_lock:
            self.stream_quality = quality
            self.active_stream_clients += 1
            self.streaming = True
            logger.info(
                "Streaming client connected (clients=%s, quality=%s)",
                self.active_stream_clients,
                quality,
            )
        return True

    def stop_streaming(self):
        """Stop streaming mode"""
        with self.stream_lock:
            if self.active_stream_clients > 0:
                self.active_stream_clients -= 1
            self.streaming = self.active_stream_clients > 0
            logger.info(
                "Streaming client disconnected (clients=%s)",
                self.active_stream_clients,
            )

    def generate_frames(self):
        """
        Generator for MJPEG streaming

        Yields:
            JPEG frame bytes in multipart format
        """
        if not self.start_streaming():
            return

        try:
            while True:
                try:
                    frame_bytes = self.capture_jpeg(self.stream_quality)

                    if frame_bytes:
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                        )
                        time.sleep(self.stream_frame_delay)
                    else:
                        # If capture fails, wait and retry
                        time.sleep(0.1)

                except GeneratorExit:
                    # Client disconnected
                    logger.info("Streaming generator closed by client")
                    break
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    break
        finally:
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
