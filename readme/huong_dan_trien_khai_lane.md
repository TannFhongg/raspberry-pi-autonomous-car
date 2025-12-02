# 🚗 Hướng dẫn triển khai Lane Detection cho Autonomous Car

## 📋 Tổng quan vấn đề

### Hiện trạng dự án của bạn:
- ✅ **Phần cứng**: Raspberry Pi 5 + Camera Module 2 + Arduino Uno
- ✅ **Độ phân giải**: 640x480 pixels
- ⚠️ **VẤN ĐỀ CHÍNH**: Code detect vạch **TRẮNG/VÀNG** nhưng thực tế là vạch **ĐEN**

### Thông số thực tế:
- **Lane width**: 38cm (khoảng cách 2 vạch kẻ)
- **Robot width**: 15cm
- **Line color**: ĐEN (trên nền sáng - nhà/trắng)
- **Camera**: Picamera2 với RGB888

---

## 🔧 Các thay đổi chính

### 1️⃣ Xử lý ảnh cho vạch đen

```python
# ❌ CŨ: Detect vạch trắng
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 30, 255])

# ✅ MỚI: Đảo ngược ảnh (vạch đen thành trắng)
gray_inverted = cv2.bitwise_not(gray)
```

**Lý do**: Thuật toán Canny và Hough hoạt động tốt hơn với đối tượng SÁNG trên nền TỐI.

### 2️⃣ Tăng cường tương phản (CLAHE)

```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced = clahe.apply(blur)
```

**Lợi ích**: Làm nổi bật vạch đen trên nền sáng không đều (ánh sáng thay đổi).

### 3️⃣ Calibration độ rộng lane

```python
# Hàm calibrate_lane_width() trong file mới
# Đo 38cm thực tế = ? pixels trên ảnh
LANE_WIDTH_PIXELS = 250  # Ví dụ, cần đo lại
```

**Cách đo**: Chụp ảnh xe đứng giữa lane → Dùng tool calibration.

### 4️⃣ Điều chỉnh tham số Hough Transform

| Tham số | Cũ | Mới | Lý do |
|---------|-----|-----|-------|
| `canny_low` | 50 | 30 | Bắt cạnh mờ hơn |
| `canny_high` | 150 | 100 | Tương tự |
| `hough_threshold` | 20 | 15 | Nhạy hơn với vạch đen |
| `min_line_length` | 30 | 25 | Detect vạch ngắn |
| `max_line_gap` | 20 | 15 | Tránh nối điểm xa |
| `blur_kernel` | 5 | 7 | Giảm nhiễu nền nhà |

### 5️⃣ Logic xử lý 1 vạch

```python
# Khi chỉ thấy vạch TRÁI
if left_lane_x is not None and right_lane_x is None:
    x_line = left_lane_x + (LANE_WIDTH_PIXELS // 2)

# Khi chỉ thấy vạch PHẢI  
if right_lane_x is not None and left_lane_x is None:
    x_line = right_lane_x - (LANE_WIDTH_PIXELS // 2)
```

**Quan trọng**: Dự đoán tâm đường khi mất 1 vạch.

---

## 📦 Cài đặt từng bước

### BƯỚC 1: Backup code cũ

```bash
cd ~/logisticsbot
cp perception/lane_detector.py perception/lane_detector_OLD.py
```

### BƯỚC 2: Copy file mới

1. **Copy `lane_detector_optimized.py`** vào `perception/`
2. **Copy `test_lane_optimized.py`** vào thư mục gốc
3. **Cập nhật `hardware_config.yaml`** với phần AI config mới

### BƯỚC 3: Cài đặt dependencies (nếu thiếu)

```bash
sudo apt-get update
sudo apt-get install -y python3-opencv
pip3 install numpy pyyaml
```

---

## 🧪 Quy trình test chi tiết

### TEST 1: Calibration (Bắt buộc - Chạy 1 lần)

```bash
cd ~/logisticsbot
python3 test_lane_optimized.py
# Chọn: 1 (CALIBRATION)
```

**Kết quả mong đợi**:
```
CALIBRATION RESULT:
  Lane Width (Real):  38 cm
  Lane Width (Pixel): 245 px  # Số này sẽ khác nhau
  Scale Factor:       0.1551 cm/px
```

**Hành động**: Cập nhật `LANE_WIDTH_PIXELS = 245` trong `lane_detector_optimized.py` dòng 47.

---

### TEST 2: So sánh 2 phương pháp trên ảnh tĩnh

```bash
python3 test_lane_optimized.py
# Chọn: 2 (TEST ẢNH TĨNH)
```

**Kết quả mong đợi**:
```
FILENAME                  | METHOD               | ERROR    | ACTION
----------------------------------------------------------------------
test_full_hd.jpg          | Hough Transform      | -15      | Đi THẲNG (^)
                          | Adaptive Threshold   | -18      | Đi THẲNG (^)
----------------------------------------------------------------------
road_curve_left.jpg       | Hough Transform      | -65      | Rẽ TRÁI  (<-)
                          | Adaptive Threshold   | -70      | Rẽ TRÁI  (<-)
```

**Kiểm tra file debug**:
- `debug_hough_test_full_hd.jpg` → Xem vạch xanh lá/xanh dương (trái/phải)
- `debug_adaptive_test_full_hd.jpg` → Xem contour vùng vạch

**Đánh giá**:
- ✅ Nếu 2 phương pháp cho kết quả tương tự (chênh < 30px) → **TỐT**
- ⚠️ Nếu chênh > 50px → Cần tune thêm tham số

---

### TEST 3: Real-time với camera (Khuyến nghị)

```bash
python3 test_lane_optimized.py
# Chọn: 3 (TEST REAL-TIME)
```

**Phím tắt**:
- `q`: Thoát
- `c`: Chụp ảnh test
- `s`: Chuyển phương pháp (Hough ↔ Adaptive)

**Kiểm tra**:
1. Đặt xe ở giữa lane → Error phải gần **0**
2. Di chuyển xe sang trái → Error **âm** (< -20)
3. Di chuyển xe sang phải → Error **dương** (> +20)

---

## 🎛️ Tuning tham số PID

### Quy trình tune:

1. **Bắt đầu với preset "ultra_safe"**:
   ```yaml
   lane_following:
     pid:
       kp: 0.5
       ki: 0.0
       kd: 0.2
   ```

2. **Test trên đường thẳng**:
   - Xe chạy **êm**, không dao động → Tăng `kp` lên 0.7
   - Xe chạy **chậm** quay → Tăng `kp` lên 0.9

3. **Test trên đường cong**:
   - Xe **không kịp rẽ** → Tăng `kp` lên 1.2
   - Xe **rẽ quá** (vượt qua) → Giảm `kp` xuống 1.0

4. **Xử lý dao động**:
   - Xe **lắc qua lắc lại** → Tăng `kd` lên 0.5
   - Xe **chậm ổn định** → Giảm `kd` xuống 0.3

5. **Xử lý sai số tích lũy** (nếu xe bị lệch dần):
   - Thêm `ki: 0.01` → Test → Tăng dần đến 0.03

### Bảng preset theo địa hình:

| Địa hình | Preset | Mô tả |
|----------|--------|-------|
| Sàn nhà bằng phẳng | `conservative` | Êm, ít dao động |
| Sân thi có gồ ghề | `balanced` | Cân bằng tốc độ/ổn định |
| Đường cong gấp khúc | `aggressive` | Phản ứng nhanh |
| Test lần đầu | `ultra_safe` | Rất chậm, an toàn |

---

## 🐛 Xử lý lỗi thường gặp

### Lỗi 1: "No lane detected" liên tục

**Nguyên nhân**: Không detect được vạch đen.

**Giải pháp**:
1. Kiểm tra ánh sáng: Nếu quá tối → Bật đèn
2. Giảm `canny_low` xuống **20**
3. Giảm `hough_threshold` xuống **10**
4. Thử phương pháp `adaptive` thay vì `hough`

```python
# Trong test_lane_optimized.py, dòng chạy detect:
error, x_line, center_x, debug = detect_line_black_adaptive(frame)
```

---

### Lỗi 2: Detect nhầm nhiễu (đồ vật, bóng đổ)

**Giải pháp**:
1. Thu hẹp ROI: Tăng `roi_top_ratio` từ 0.4 → **0.5**
2. Tăng `min_line_length` lên **35**
3. Tăng `min_slope` lên **0.5** (loại đường ngang)

---

### Lỗi 3: Xe dao động mạnh (oscillation)

**Nguyên nhân**: PID không ổn định.

**Giải pháp**:
1. Giảm `kp` xuống **0.7**
2. Tăng `kd` lên **0.5**
3. Giảm `base_speed` xuống **100**

---

### Lỗi 4: Mất 1 vạch → Xe lạc hướng

**Kiểm tra**: `LANE_WIDTH_PIXELS` đã calibrate chưa?

```python
# Trong lane_detector_optimized.py, dòng 47
LANE_WIDTH_PIXELS = 245  # ⚠️ Phải khớp với calibration
```

---

## 📊 Đánh giá chất lượng detection

### Metrics để đánh giá:

1. **Detection rate**: % frames detect được lane
   - **Mục tiêu**: > 95%

2. **Error stability**: Độ ổn định của error
   - **Tốt**: Error biến thiên < ±10px giữa các frame
   - **Xấu**: Nhảy lung tung ±50px

3. **Response time**: Thời gian phản ứng khi vào cua
   - **Tốt**: Xe bắt đầu rẽ trong vòng 0.3s

4. **Lane keeping accuracy**: Sai số trung bình
   - **Mục tiêu**: < ±15px (tương đương ±2.3cm với scale 0.155)

### Tool debug:

Thêm logging vào code chính:

```python
# Trong file điều khiển chính
import csv

with open('lane_log.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Frame', 'Error', 'Action', 'Speed'])
    
    for frame_count in range(1000):
        error, x_line, center_x, _ = detect_line(frame)
        action = calculate_action(error)  # Hàm PID của bạn
        
        writer.writerow([frame_count, error, action, current_speed])
```

Sau đó plot graph bằng Excel/Python để phân tích.

---

## 🚀 Tối ưu hóa nâng cao

### 1. Kalman Filter (Làm mượt error)

```python
# Thêm vào đầu file
class KalmanFilter1D:
    def __init__(self, process_variance=1e-5, measurement_variance=1e-1):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = 0
        self.error = 1
    
    def update(self, measurement):
        # Prediction
        prediction = self.estimate
        prediction_error = self.error + self.process_variance
        
        # Update
        kalman_gain = prediction_error / (prediction_error + self.measurement_variance)
        self.estimate = prediction + kalman_gain * (measurement - prediction)
        self.error = (1 - kalman_gain) * prediction_error
        
        return self.estimate

# Sử dụng:
kf = KalmanFilter1D()
error_filtered = kf.update(error)
```

### 2. Moving Average (Đơn giản hơn)

```python
error_history = []

def smooth_error(error, window=5):
    error_history.append(error)
    if len(error_history) > window:
        error_history.pop(0)
    return sum(error_history) / len(error_history)
```

### 3. Hybrid Detection (Kết hợp 2 phương pháp)

```python
error_hough, _, _, _ = detect_line(frame)
error_adaptive, _, _, _ = detect_line_black_adaptive(frame)

# Nếu 2 phương pháp cho kết quả gần nhau → Tin tưởng
if abs(error_hough - error_adaptive) < 30:
    error_final = (error_hough + error_adaptive) / 2
else:
    # Chọn phương pháp có confidence cao hơn
    error_final = error_hough  # Hoặc logic phức tạp hơn
```

---

## 📋 Checklist trước khi thi

- [ ] Đã chạy calibration và cập nhật `LANE_WIDTH_PIXELS`
- [ ] Test trên đường thẳng: Error < ±20px
- [ ] Test trên đường cong: Xe rẽ đúng hướng
- [ ] PID đã tune: Không dao động, phản ứng nhanh
- [ ] Backup code cũ và config
- [ ] Test trên nền trắng (gần với sân thi nhất)
- [ ] Kiểm tra ánh sáng: Camera không bị chói/tối
- [ ] Kiểm tra pin/nguồn: Đủ điện cho 5-10 phút chạy
- [ ] Có file log để debug khi lỗi

---

## 📞 Các tình huống xử lý khi thi

### Tình huống 1: Xe mất lane đột ngột

```python
# Trong file điều khiển chính
lane_lost_count = 0

if error == 0 and x_line == center_x:  # Không detect được
    lane_lost_count += 1
    
    if lane_lost_count > 10:  # Mất 10 frames liên tục
        # Dừng xe
        stop_robot()
        print("⚠️ LANE LOST - STOPPED")
else:
    lane_lost_count = 0
```

### Tình huống 2: Xe chạy ra khỏi line

→ **Kiểm tra lại calibration** và giảm tốc độ.

### Tình huống 3: Ánh sáng thay đổi đột ngột

→ Dùng phương pháp **Adaptive Threshold** thay vì Hough.

---

## 🎯 Kết luận

### Ưu tiên cao nhất:

1. **CALIBRATION** đúng `LANE_WIDTH_PIXELS`
2. **TEST** đầy đủ 3 bước (Calibration → Static → Real-time)
3. **TUNE PID** theo địa hình thực tế
4. **BACKUP** code trước khi thi

### Điểm mạnh của giải pháp:

✅ Xử lý đúng vạch ĐEN (đảo màu + CLAHE)  
✅ Có 2 phương pháp dự phòng (Hough + Adaptive)  
✅ Logic xử lý 1 vạch thông minh  
✅ Tool calibration tự động  
✅ Logging chi tiết để debug  

### Lưu ý cuối:

- **Nền nhà khác nền thi**: Test trên cả 2 để chắc chắn
- **Vạch đen mờ**: Giảm threshold Canny/Hough
- **Nhiễu cao**: Tăng blur_kernel và dùng CLAHE

**Chúc bạn thành công! 🚗💨**