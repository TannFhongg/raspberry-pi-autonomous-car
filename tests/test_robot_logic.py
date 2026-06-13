import unittest
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import yaml

import dashboard_server
import main
import tune_lane_web
from control import robot_controller as robot_module
from perception import lane_detector
from perception.camera_manager import CameraManager, StreamingCameraManager


class FakeDriver:
    def __init__(self):
        self.left_speed = 0
        self.right_speed = 0
        self.commands = []
        self.cleaned = False

    def set_motors(self, left_speed, right_speed):
        self.left_speed = left_speed
        self.right_speed = right_speed
        self.commands.append(("set_motors", left_speed, right_speed))

    def turn_left(self, speed):
        self.set_motors(-speed, speed)
        self.commands.append(("turn_left", speed))

    def turn_right(self, speed):
        self.set_motors(speed, -speed)
        self.commands.append(("turn_right", speed))

    def stop(self):
        self.left_speed = 0
        self.right_speed = 0
        self.commands.append(("stop",))

    def get_speeds(self):
        return self.left_speed, self.right_speed

    def cleanup(self):
        self.cleaned = True


class FakeIMU:
    connected = False

    def start(self):
        pass

    def stop(self):
        pass


def make_robot_controller(config=None):
    driver = FakeDriver()
    config = config or {"safety": {"timeout": 5.0}, "robot": {"default_speed": 180}}
    with mock.patch.object(robot_module, "IMUSensorFusion", FakeIMU):
        controller = robot_module.RobotController(driver, config)
    return controller, driver


class FakeArrayCamera:
    def __init__(self, frame):
        self.frame = frame
        self.capture_count = 0
        self.stopped = False

    def capture_array(self):
        self.capture_count += 1
        return self.frame.copy()

    def stop(self):
        self.stopped = True


class DummyStreamingCamera(StreamingCameraManager):
    def __init__(self):
        super().__init__({"sensors": {"camera": {"framerate": 30}}})
        self.running = True
        self.camera = FakeArrayCamera(np.zeros((6, 4), dtype=np.uint8))
        self.jpeg_count = 0

    def start(self):
        self.running = True
        return True

    def capture_jpeg(self, quality=80):
        self.jpeg_count += 1
        return b"\xff\xd8fake-jpeg\xff\xd9"


class CameraManagerTests(unittest.TestCase):
    def test_capture_jpeg_populates_buffer_when_no_control_loop_is_running(self):
        frame_yuv = np.full((6, 4), 128, dtype=np.uint8)
        frame_yuv[:4, :] = 90
        fake_camera = FakeArrayCamera(frame_yuv)
        manager = CameraManager(
            {"sensors": {"camera": {"resolution": [4, 4], "picamera2": {"format": "YUV420"}}}}
        )
        manager.camera = fake_camera
        manager.running = True

        jpeg = manager.capture_jpeg(quality=500)

        self.assertIsNotNone(jpeg)
        self.assertTrue(jpeg.startswith(b"\xff\xd8"))
        self.assertEqual(fake_camera.capture_count, 1)
        with manager.latest_frame_lock:
            self.assertIsNotNone(manager.latest_frame)
            self.assertGreater(manager.latest_frame_time, 0)

    def test_stream_generators_are_per_client(self):
        camera = DummyStreamingCamera()

        gen1 = camera.generate_frames()
        gen2 = camera.generate_frames()
        try:
            self.assertIn(b"Content-Type: image/jpeg", next(gen1))
            self.assertIn(b"Content-Type: image/jpeg", next(gen2))
            self.assertEqual(camera.active_stream_clients, 2)

            gen1.close()
            self.assertEqual(camera.active_stream_clients, 1)
            self.assertTrue(camera.streaming)
            self.assertIn(b"Content-Type: image/jpeg", next(gen2))
        finally:
            gen2.close()

        self.assertEqual(camera.active_stream_clients, 0)
        self.assertFalse(camera.streaming)


class LaneDetectorTests(unittest.TestCase):
    def test_yuv420_detection_uses_actual_frame_width(self):
        frame_yuv = np.full((90, 80), 128, dtype=np.uint8)
        with mock.patch.object(lane_detector.cv2, "HoughLinesP", return_value=None):
            error, x_line, center_x, _ = lane_detector.detect_line(
                frame_yuv,
                {"blur_kernel": 1},
                debug=False,
            )

        self.assertEqual(center_x, 40)
        self.assertEqual(x_line, 40)
        self.assertEqual(error, 999)

    def test_lane_width_pixels_config_changes_single_lane_estimate(self):
        frame_bgr = np.full((60, 80, 3), 255, dtype=np.uint8)
        hough_line = np.array([[[20, 50, 30, 40]]], dtype=np.int32)

        with mock.patch.object(lane_detector.cv2, "HoughLinesP", return_value=hough_line):
            error_small, x_small, _, _ = lane_detector.detect_line(
                frame_bgr,
                {"lane_width_pixels": 20, "blur_kernel": 1, "camera_offset": 0},
                debug=False,
            )

        with mock.patch.object(lane_detector.cv2, "HoughLinesP", return_value=hough_line):
            error_large, x_large, _, _ = lane_detector.detect_line(
                frame_bgr,
                {"lane_width_pixels": 60, "blur_kernel": 1, "camera_offset": 0},
                debug=False,
            )

        self.assertNotEqual(x_small, x_large)
        self.assertNotEqual(error_small, error_large)
        self.assertLess(x_small, x_large)


class RobotControllerTests(unittest.TestCase):
    def test_speed_setter_clamps_and_limits_safe_motor_output(self):
        controller, driver = make_robot_controller()
        try:
            self.assertEqual(controller.set_speed(300), 255)
            self.assertEqual(controller.current_speed, 255)

            self.assertEqual(controller.set_speed(100), 100)
            self.assertTrue(controller.safe_set_motors(200, -150))
            self.assertEqual(driver.get_speeds(), (100, -100))
        finally:
            controller.cleanup()

    def test_watchdog_timeout_stops_active_motors_and_preserves_emergency(self):
        controller, driver = make_robot_controller({"safety": {"timeout": 5.0}})
        try:
            controller.current_mode = "auto"
            driver.set_motors(60, 60)
            controller.last_command_time = 100.0

            self.assertTrue(controller.check_watchdog_timeout(now=106.0))
            self.assertEqual(driver.get_speeds(), (0, 0))
            self.assertEqual(controller.current_mode, "idle")
            self.assertEqual(controller.current_state, "WATCHDOG TIMEOUT - STOPPED")

            controller.current_mode = "auto"
            controller.current_state = "EMERGENCY STOP"
            controller.emergency_stopped = True
            controller.last_command_time = 100.0

            self.assertFalse(controller.check_watchdog_timeout(now=200.0))
            self.assertEqual(controller.current_state, "EMERGENCY STOP")
        finally:
            controller.cleanup()

    def test_smart_turn_is_blocked_by_emergency_stop(self):
        controller, driver = make_robot_controller()
        try:
            controller.emergency_stop()

            self.assertFalse(controller.smart_turn(90))
            self.assertNotIn(("turn_left", 90), driver.commands)
            self.assertNotIn(("turn_right", 90), driver.commands)
        finally:
            controller.cleanup()


class FakeRobot:
    def __init__(self):
        self.current_mode = "idle"
        self.current_state = "STANDBY"
        self.speed = 180
        self.stopped = False

    def set_mode(self, mode):
        if mode not in ["auto", "follow", "idle"]:
            return False
        self.current_mode = mode
        self.current_state = {
            "auto": "AUTO MODE",
            "follow": "FOLLOW MODE",
            "idle": "STANDBY",
        }[mode]
        return True

    def stop(self):
        self.stopped = True
        return True

    def set_speed(self, speed):
        self.speed = max(0, min(255, int(speed)))
        return self.speed

    def get_state(self):
        return {
            "state": self.current_state,
            "speed": self.speed,
            "left_motor_speed": 0,
            "right_motor_speed": 0,
        }


class FakeModeController:
    def __init__(self, start_result=True):
        self.start_result = start_result
        self.running = False
        self.stop_count = 0
        self.speed = None
        self.colors = []

    def start(self):
        self.running = self.start_result
        return self.start_result

    def stop(self):
        self.running = False
        self.stop_count += 1

    def set_speed(self, speed):
        self.speed = speed
        return speed

    def set_target_color(self, color):
        self.colors.append(color)


class MainRouteTests(unittest.TestCase):
    def setUp(self):
        self.saved_globals = {
            "robot_controller": main.robot_controller,
            "auto_controller": main.auto_controller,
            "follow_controller": main.follow_controller,
            "motor_driver": main.motor_driver,
            "log_message": main.log_message,
        }
        self.emit_patcher = mock.patch.object(main.socketio, "emit", lambda *args, **kwargs: None)
        self.emit_patcher.start()
        main.log_message = lambda *args, **kwargs: None
        main.motor_driver = object()
        main.app.testing = True
        self.client = main.app.test_client()

    def tearDown(self):
        self.emit_patcher.stop()
        for name, value in self.saved_globals.items():
            setattr(main, name, value)

    def test_follow_distance_route_and_frontend_references_are_removed(self):
        rules = {rule.rule for rule in main.app.url_map.iter_rules()}
        self.assertNotIn("/set_follow_distance", rules)

        for relative_path in ["static/js/app.js", "templates/index.html"]:
            content = Path(relative_path).read_text(encoding="utf-8")
            self.assertNotIn("/set_follow_distance", content)
            self.assertNotIn("followDistance", content)

    def test_follow_color_support_rejects_orange(self):
        main.robot_controller = FakeRobot()
        main.follow_controller = FakeModeController()

        ok_response = self.client.get("/set_follow_color?color=red")
        self.assertEqual(ok_response.status_code, 200)
        self.assertEqual(ok_response.get_json()["color"], "red")
        self.assertEqual(main.follow_controller.colors, ["red"])

        bad_response = self.client.get("/set_follow_color?color=orange")
        self.assertEqual(bad_response.status_code, 400)
        self.assertEqual(bad_response.get_json()["status"], "error")

    def test_set_speed_propagates_to_mode_controllers(self):
        main.robot_controller = FakeRobot()
        main.auto_controller = FakeModeController()
        main.follow_controller = FakeModeController()

        response = self.client.get("/set_speed?value=123")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["speed"], 123)
        self.assertEqual(main.robot_controller.speed, 123)
        self.assertEqual(main.auto_controller.speed, 123)
        self.assertEqual(main.follow_controller.speed, 123)

        bad_response = self.client.get("/set_speed?value=300")
        self.assertEqual(bad_response.status_code, 400)

    def test_set_mode_start_failure_returns_503_and_rolls_back_to_idle(self):
        main.robot_controller = FakeRobot()
        main.auto_controller = FakeModeController(start_result=False)
        main.follow_controller = FakeModeController()

        response = self.client.get("/set_mode?mode=auto")
        data = response.get_json()

        self.assertEqual(response.status_code, 503)
        self.assertEqual(data["status"], "error")
        self.assertEqual(data["mode"], "idle")
        self.assertEqual(main.robot_controller.current_mode, "idle")
        self.assertFalse(main.auto_controller.running)
        self.assertGreaterEqual(main.auto_controller.stop_count, 1)


class DashboardValidationTests(unittest.TestCase):
    def setUp(self):
        self.original_lane_params = dashboard_server.lane_params.copy()
        dashboard_server.app.testing = True
        self.client = dashboard_server.app.test_client()

    def tearDown(self):
        dashboard_server.lane_params = self.original_lane_params.copy()

    def test_blur_kernel_validation_normalizes_even_values(self):
        self.assertEqual(dashboard_server.validate_lane_param("blur_kernel", 4), 5)
        self.assertEqual(dashboard_server.validate_lane_param("blur_kernel", 16), 15)
        self.assertEqual(dashboard_server.validate_lane_param("blur_kernel", 0), 1)

        response = self.client.post(
            "/update_param",
            json={"name": "blur_kernel", "value": 4},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["value"], 5)
        self.assertEqual(dashboard_server.lane_params["blur_kernel"], 5)

    def test_lane_param_validation_rejects_invalid_payloads(self):
        bad_name = self.client.post(
            "/update_param",
            json={"name": "not_real", "value": 1},
        )
        self.assertEqual(bad_name.status_code, 400)

        bad_value = self.client.post(
            "/update_param",
            json={"name": "blur_kernel", "value": "abc"},
        )
        self.assertEqual(bad_value.status_code, 400)

        bad_json = self.client.post("/update_param", data="not json")
        self.assertEqual(bad_json.status_code, 400)

    def test_dashboard_uses_only_rendered_slider_names(self):
        response = self.client.get("/")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("const tunableParamNames", html)
        self.assertNotIn("Object.keys(defaultParams)", html)

    def test_save_params_updates_hardware_config_lane_section(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "hardware_config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "ai:",
                        "  lane_detection:",
                        "    canny_low: 80",
                        "    blur_kernel: 7",
                        "other:",
                        "  unchanged: true",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            dashboard_server.lane_params.update({
                "canny_low": 123,
                "blur_kernel": 9,
            })
            dashboard_server.save_tunable_lane_params_to_config(config_path)

            saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["ai"]["lane_detection"]["canny_low"], 123)
            self.assertEqual(saved["ai"]["lane_detection"]["blur_kernel"], 9)
            self.assertTrue(saved["other"]["unchanged"])


class TuneLaneWebTests(unittest.TestCase):
    def setUp(self):
        self.original_config_path = tune_lane_web.config_path
        self.original_params = tune_lane_web.current_params.copy()
        tune_lane_web.app.testing = True
        self.client = tune_lane_web.app.test_client()

    def tearDown(self):
        tune_lane_web.config_path = self.original_config_path
        tune_lane_web.current_params = self.original_params.copy()

    def test_load_lane_params_prefers_ai_and_falls_back_to_legacy(self):
        params = tune_lane_web.load_lane_params({
            "ai": {"lane_detection": {"canny_low": 91}},
            "lane_following": {"lane_detection": {"canny_low": 12}},
        })
        self.assertEqual(params["canny_low"], 91)

        legacy_params = tune_lane_web.load_lane_params({
            "lane_following": {"lane_detection": {"canny_low": 12}},
        })
        self.assertEqual(legacy_params["canny_low"], 12)

    def test_update_params_validates_payload_and_normalizes_blur_kernel(self):
        bad_json = self.client.post("/update_params", data="not json")
        self.assertEqual(bad_json.status_code, 400)

        bad_name = self.client.post("/update_params", json={"not_real": 1})
        self.assertEqual(bad_name.status_code, 400)

        bad_value = self.client.post("/update_params", json={"canny_low": "abc"})
        self.assertEqual(bad_value.status_code, 400)

        response = self.client.post("/update_params", json={"blur_kernel": 4})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["success"])
        self.assertEqual(response.get_json()["params"]["blur_kernel"], 5)
        self.assertEqual(tune_lane_web.current_params["blur_kernel"], 5)

    def test_save_config_writes_ai_and_mirrors_existing_legacy_section(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "hardware_config.yaml"
            config_path.write_text(
                yaml.safe_dump({
                    "ai": {"lane_detection": {"canny_low": 1}},
                    "lane_following": {"lane_detection": {"canny_low": 2}},
                    "other": {"unchanged": True},
                }),
                encoding="utf-8",
            )

            tune_lane_web.config_path = str(config_path)
            tune_lane_web.current_params = tune_lane_web.DEFAULT_PARAMS.copy()
            tune_lane_web.current_params.update({
                "canny_low": 123,
                "canny_high": 234,
                "roi_top_ratio": 0.4,
                "hough_threshold": 55,
                "blur_kernel": 9,
            })

            response = self.client.post("/save_config")
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.get_json()["success"])

            saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["ai"]["lane_detection"]["canny_low"], 123)
            self.assertEqual(saved["lane_following"]["lane_detection"]["canny_low"], 123)
            self.assertTrue(saved["other"]["unchanged"])

    def test_frame_to_bgr_converts_yuv420_to_image_shape(self):
        frame_yuv = np.full((90, 80), 128, dtype=np.uint8)
        frame_yuv[:60, :] = 90

        frame_bgr = tune_lane_web.frame_to_bgr(frame_yuv, "YUV420")

        self.assertEqual(frame_bgr.shape, (60, 80, 3))


if __name__ == "__main__":
    unittest.main()
