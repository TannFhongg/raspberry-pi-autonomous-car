# 🤖 Autonomous_Car - Hệ Thống Xe Tự Hành

Hệ thống điều khiển xe robot tự hành sử dụng Raspberry Pi và Arduino, tích hợp xử lý ảnh để theo dõi làn đường và nhận diện biển báo giao thông.

## 📋 Tổng Quan

Autonomous_Car là dự án xe tự hành mini với các tính năng:

- **Chế độ Manual**: Điều khiển thủ công qua web dashboard
- **Chế độ Auto**: Tự động theo dõi làn đường (lane following) với PID controller
- **Chế độ Follow**: Theo dõi đối tượng theo màu sắc
- **Nhận diện biển báo**: Dừng, rẽ trái/phải, đèn giao thông
- **Visual Odometry**: Ước tính vị trí từ camera
- **IMU Sensor Fusion**: Điều khiển xoay chính xác

## 🛠️ Phần Cứng

| Thành phần | Mô tả |
|------------|-------|
| Raspberry Pi 5 | Bộ xử lý chính |
| Arduino Uno/Nano | Điều khiển motor và cảm biến |
| L298N Motor Driver | Điều khiển 2 động cơ DC |
| Raspberry Pi Camera V2 | Camera 8MP (1640x1232) |
| HC-SR04 | Cảm biến siêu âm đo khoảng cách |
| MPU6050 (tùy chọn) | IMU sensor cho smart turn |

## 📁 Cấu Trúc Dự Án

```
├── main.py                 # Entry point - Flask web server
├── config/
│   └── hardware_config.yaml # Cấu hình phần cứng và PID
├── control/
│   ├── robot_controller.py  # Điều khiển robot chính
│   └── pid_controller.py    # PID controller
├── perception/
│   ├── lane_detector.py     # Phát hiện làn đường
│   ├── object_detector.py   # Nhận diện biển báo (YOLO)
│   ├── camera_manager.py    # Quản lý Picamera2
│   ├── visual_odometry.py   # Ước tính vị trí
│   └── imu_sensor_fusion.py # Xử lý IMU
├── drivers/
│   └── motor/               # Driver điều khiển motor
├── arduino_firmware/
│   └── arduino_firmware.ino # Code Arduino
├── static/                  # CSS, JS cho web
├── templates/               # HTML templates
└── tools/                   # Công cụ hỗ trợ
```

## 🚀 Cài Đặt

### 1. Clone repository
```bash
git clone <repository-url>
cd LogisticsBot
```

### 2. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 3. Nạp firmware Arduino
- Mở `arduino_firmware/arduino_firmware.ino` bằng Arduino IDE
- Upload lên Arduino Uno/Nano

### 4. Cấu hình
Chỉnh sửa `config/hardware_config.yaml`:
- Cổng serial Arduino (`/dev/ttyACM0` hoặc `/dev/ttyUSB0`)
- Thông số PID
- Cấu hình camera

## 💻 Sử Dụng

### Khởi động server
```bash
python main.py
```

### Truy cập dashboard
Mở trình duyệt: `http://<raspberry-pi-ip>:5000`

### Các chế độ hoạt động

| Chế độ | Mô tả |
|--------|-------|
| Manual | Điều khiển bằng nút bấm trên dashboard |
| Auto | Tự động theo làn đường đen trên nền trắng |
| Follow | Theo dõi đối tượng theo màu (đỏ/xanh/vàng) |

## ⚙️ Cấu Hình PID

Các preset có sẵn trong `hardware_config.yaml`:

| Preset | Kp | Ki | Kd | Tốc độ | Mô tả |
|--------|----|----|----|----|-------|
| ultra_safe | 0.5 | 0.0 | 0.2 | 100 | Test ban đầu |
| balanced | 0.3 | 0.0 | 0.05 | 90 | Khuyến nghị |
| aggressive | 1.5 | 0.03 | 0.6 | 150 | Đường cong gấp |

## 📖 Tài Liệu Chi Tiết

- [Hướng dẫn đấu nối Arduino Uno](readme/ARDUINO_UNO_WIRING.md)
- [Hướng dẫn cài đặt đầy đủ](readme/COMPLETE_SETUP_GUIDE.md)
- [Cấu hình PID Lane Following](readme/PID_LANE_FOLLOWING_SETUP.md)
- [Hướng dẫn IMU](readme/HuongDan_IMU.md)

## 🔧 Calibration

Trước khi chạy Auto mode, cần calibrate lane width:
```bash
python test_lane_optimized.py
```

## 📝 API Endpoints

| Endpoint | Mô tả |
|----------|-------|
| `/` | Dashboard chính |
| `/video_feed` | Stream video camera |
| `/debug_feed` | Stream debug (lane detection) |
| `/set_mode?mode=auto` | Đổi chế độ |
| `/forward`, `/backward`, `/left`, `/right` | Điều khiển |
| `/stop`, `/emergency_stop` | Dừng |
| `/set_speed?value=150` | Đặt tốc độ |

## 📄 License

Vo Van Tuan 
