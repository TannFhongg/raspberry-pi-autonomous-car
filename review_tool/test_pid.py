"""
Test PID Controller - Kiểm tra thuật toán PID
Chạy: python review_tool/test_pid.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import matplotlib.pyplot as plt
from control.pid_controller import PIDController
from utils.config_loader import load_config


def test_pid_step_response():
    """Test PID với step response"""
    print("=" * 60)
    print("PID TEST - STEP RESPONSE")
    print("=" * 60)
    
    # Load config
    try:
        config = load_config('config/hardware_config.yaml')
        pid_cfg = config.get('lane_following', {}).get('pid', {})
        print("✅ Loaded PID config from hardware_config.yaml")
    except:
        pid_cfg = {'kp': 0.45, 'ki': 0.002, 'kd': 0.08}
        print("⚠️  Dùng PID config mặc định")
    
    print(f"\nPID Gains:")
    print(f"  Kp: {pid_cfg.get('kp', 0.45)}")
    print(f"  Ki: {pid_cfg.get('ki', 0.002)}")
    print(f"  Kd: {pid_cfg.get('kd', 0.08)}")
    
    # Initialize PID
    pid = PIDController(
        kp=pid_cfg.get('kp', 0.45),
        ki=pid_cfg.get('ki', 0.002),
        kd=pid_cfg.get('kd', 0.08),
        output_min=-255,
        output_max=255
    )
    
    # Simulate step input (lane error)
    print("\n" + "=" * 60)
    print("SIMULATION")
    print("=" * 60)
    print("Mô phỏng: Xe lệch 100px sang trái → PID điều chỉnh về 0")
    
    # Test data
    dt = 0.03  # 30ms cycle time
    errors = []
    outputs = []
    p_terms = []
    i_terms = []
    d_terms = []
    times = []
    
    # Simulate for 3 seconds
    current_error = 100.0  # Initial error
    sim_time = 0
    
    for i in range(100):
        # Compute PID
        output = pid.compute(current_error, dt)
        components = pid.get_components()
        
        # Store data
        errors.append(current_error)
        outputs.append(output)
        p_terms.append(components['p'])
        i_terms.append(components['i'])
        d_terms.append(components['d'])
        times.append(sim_time)
        
        # Simple plant model: error reduces based on output
        # (trong thực tế, output điều khiển motor → robot xoay → error thay đổi)
        current_error = current_error - (output * 0.01)
        
        sim_time += dt
    
    # Plot results
    print("\nVẽ đồ thị kết quả...")
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    
    # Error plot
    axes[0].plot(times, errors, 'r-', linewidth=2)
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0].set_ylabel('Error (px)')
    axes[0].set_title('PID Step Response Test')
    axes[0].grid(True, alpha=0.3)
    
    # Output plot
    axes[1].plot(times, outputs, 'b-', linewidth=2)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1].set_ylabel('Control Output')
    axes[1].grid(True, alpha=0.3)
    
    # PID components plot
    axes[2].plot(times, p_terms, 'g-', label='P', linewidth=2)
    axes[2].plot(times, i_terms, 'b-', label='I', linewidth=2)
    axes[2].plot(times, d_terms, 'r-', label='D', linewidth=2)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('PID Components')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = 'review_tool/test_pid_result.png'
    plt.savefig(output_path)
    print(f"✅ Đã lưu đồ thị: {output_path}")
    
    # Analyze results
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    # Settling time (time to reach ±5px)
    settling_idx = next((i for i, e in enumerate(errors) if abs(e) < 5), len(errors))
    settling_time = times[settling_idx] if settling_idx < len(times) else times[-1]
    
    # Overshoot
    min_error = min(errors)
    overshoot = abs(min_error) if min_error < 0 else 0
    
    # Steady state error
    steady_error = abs(errors[-1])
    
    print(f"Initial Error:      {errors[0]:.1f} px")
    print(f"Settling Time:      {settling_time:.2f} s (to ±5px)")
    print(f"Overshoot:          {overshoot:.1f} px")
    print(f"Steady State Error: {steady_error:.2f} px")
    print(f"Final Error:        {errors[-1]:.1f} px")
    
    # Evaluation
    if settling_time < 1.0 and steady_error < 2:
        print("\n✅ PID Performance: EXCELLENT")
    elif settling_time < 2.0 and steady_error < 5:
        print("\n✅ PID Performance: GOOD")
    else:
        print("\n⚠️  PID Performance: NEEDS TUNING")


def test_pid_different_errors():
    """Test PID với nhiều mức error khác nhau"""
    print("\n" + "=" * 60)
    print("PID TEST - DIFFERENT ERROR LEVELS")
    print("=" * 60)
    
    try:
        config = load_config('config/hardware_config.yaml')
        pid_cfg = config.get('lane_following', {}).get('pid', {})
    except:
        pid_cfg = {'kp': 0.45, 'ki': 0.002, 'kd': 0.08}
    
    pid = PIDController(
        kp=pid_cfg.get('kp', 0.45),
        ki=pid_cfg.get('ki', 0.002),
        kd=pid_cfg.get('kd', 0.08),
        output_min=-255,
        output_max=255
    )
    
    # Test với các error level khác nhau
    test_errors = [10, 30, 50, 80, 100]
    dt = 0.03
    
    print("\nTest PID output với các error level:")
    print(f"{'Error':>10} | {'P':>10} | {'I':>10} | {'D':>10} | {'Output':>10}")
    print("-" * 60)
    
    for error in test_errors:
        pid.reset()
        output = pid.compute(error, dt)
        comp = pid.get_components()
        
        print(f"{error:>10} | {comp['p']:>10.1f} | {comp['i']:>10.1f} | "
              f"{comp['d']:>10.1f} | {output:>10.1f}")
    
    print("\n✅ Test hoàn tất")


def main():
    print("\n" + "🔧 " * 20)
    print("PID CONTROLLER TEST SUITE")
    print("🔧 " * 20)
    
    print("\nChọn test mode:")
    print("  1 - Step Response Test (đồ thị)")
    print("  2 - Error Level Test (bảng)")
    print("  3 - Cả hai")
    print("  q - Quit")
    print()
    
    choice = input("Nhập lựa chọn: ").strip().lower()
    
    if choice == '1':
        test_pid_step_response()
    elif choice == '2':
        test_pid_different_errors()
    elif choice == '3':
        test_pid_step_response()
        test_pid_different_errors()
    elif choice == 'q':
        print("Goodbye!")
    else:
        print("Lựa chọn không hợp lệ!")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test bị ngắt")
