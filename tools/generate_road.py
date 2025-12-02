import cv2
import numpy as np
import os

def create_road_image(filename, mode='straight'):
    # 1. Tạo nền ĐEN (giả lập đường nhựa) - Kích thước 320x240
    # Nếu bạn dùng nền trắng vạch đen, đổi (0,0,0) thành (255,255,255) 
    # và màu vẽ thành (0,0,0)
    width, height = 320, 240
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Màu vạch kẻ (Trắng)
    color = (255, 255, 255) 
    thickness = 5
    
    # Tọa độ cơ bản
    bottom_y = height
    top_y = int(height * 0.6) # Đường chân trời giả định
    
    center_bottom = width // 2
    lane_width_bottom = 200 # Độ rộng đường ở đáy ảnh
    lane_width_top = 80     # Độ rộng đường ở xa (nhỏ hơn do phối cảnh)
    
    if mode == 'straight':
        # --- ĐƯỜNG THẲNG (Có phối cảnh) ---
        # Vạch Trái
        pt1_l = (center_bottom - lane_width_bottom // 2, bottom_y)
        pt2_l = (center_bottom - lane_width_top // 2, top_y)
        cv2.line(img, pt1_l, pt2_l, color, thickness)
        
        # Vạch Phải
        pt1_r = (center_bottom + lane_width_bottom // 2, bottom_y)
        pt2_r = (center_bottom + lane_width_top // 2, top_y)
        cv2.line(img, pt1_r, pt2_r, color, thickness)
        
    elif mode == 'left':
        # --- CUA TRÁI ---
        # Dịch chuyển tâm về phía trái
        shift = 80
        
        # Vạch Trái
        pt1_l = (center_bottom - lane_width_bottom // 2, bottom_y)
        pt2_l = (center_bottom - lane_width_top // 2 - shift, top_y)
        cv2.line(img, pt1_l, pt2_l, color, thickness)
        
        # Vạch Phải
        pt1_r = (center_bottom + lane_width_bottom // 2, bottom_y)
        pt2_r = (center_bottom + lane_width_top // 2 - shift, top_y)
        cv2.line(img, pt1_r, pt2_r, color, thickness)

    elif mode == 'right':
        # --- CUA PHẢI ---
        # Dịch chuyển tâm về phía phải
        shift = 80
        
        # Vạch Trái
        pt1_l = (center_bottom - lane_width_bottom // 2, bottom_y)
        pt2_l = (center_bottom - lane_width_top // 2 + shift, top_y)
        cv2.line(img, pt1_l, pt2_l, color, thickness)
        
        # Vạch Phải
        pt1_r = (center_bottom + lane_width_bottom // 2, bottom_y)
        pt2_r = (center_bottom + lane_width_top // 2 + shift, top_y)
        cv2.line(img, pt1_r, pt2_r, color, thickness)

    # Lưu ảnh
    cv2.imwrite(filename, img)
    print(f"✓ Đã tạo: {filename}")

# Chạy tạo 3 ảnh
print("Đang tạo bộ ảnh test giả lập...")
create_road_image('road_straight.jpg', 'straight')
create_road_image('road_curve_left.jpg', 'left')
create_road_image('road_curve_right.jpg', 'right')
print("Hoàn tất!")


