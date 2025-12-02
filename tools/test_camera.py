from perception.camera_manager import CameraManager
from utils.config_loader import load_config

config = load_config('config/hardware_config.yaml')
camera = CameraManager(config)

if camera.start():
    print("✓ Camera started")
    
    frame = camera.capture_frame()
    print(f"✓ Frame: {frame.shape}")
    
    camera.stop()
    print("✓ Test complete")