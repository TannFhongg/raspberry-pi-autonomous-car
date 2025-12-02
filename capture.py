import time
import os
from picamera2 import Picamera2

def chup_anh_chat_luong_cao(ten_file="anh_pi5.jpg"):
    # Khởi tạo đối tượng camera
    picam2 = Picamera2()

    try:
        print("Đang khởi động camera...")
        
        # 1. Cấu hình cho Chụp Ảnh Tĩnh (Still Capture)
        # Camera V2 có độ phân giải max là 3280 x 2464
        # Chúng ta dùng create_still_configuration để ưu tiên chất lượng
        config = picam2.create_still_configuration(
            main={"size": (1640, 1232), "format": "RGB888"}
        ) 
        picam2.configure(config)

        # 2. Bắt đầu Camera
        picam2.start()
        
        # 3. Thời gian "Warm-up" (Làm nóng)
        # Rất quan trọng để Auto White Balance (AWB) và Auto Exposure (AE) ổn định
        print("Đang cân bằng sáng (2 giây)...")
        time.sleep(2)

        # 4. Chụp ảnh
        # Bạn có thể thêm các options như 'quality' nếu lưu dạng JPG
        picam2.capture_file(ten_file)
        
        print(f"Đã lưu ảnh tại: {os.path.abspath(ten_file)}")

    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")

    finally:
        # 5. Dọn dẹp tài nguyên
        # Khối finally này đảm bảo camera luôn được tắt, dù có lỗi hay không
        picam2.stop()
        picam2.close()
        print("Đã tắt camera an toàn.")

# Chạy hàm
if __name__ == "__main__":
    chup_anh_chat_luong_cao("test_full_hd.jpg")