import gymnasium as gym
import numpy as np
import time

def main():
    # Khởi tạo môi trường Pendulum với render_mode="human" để hiển thị cửa sổ trực quan
    env = gym.make("Pendulum-v1", render_mode="human")
    
    # In thông tin chi tiết về Không gian trạng thái và Không gian hành động
    print("=" * 50)
    print("CHI TIẾT MÔI TRƯỜNG PENDULUM-V1")
    print("=" * 50)
    print(f"Không gian quan sát (Observation Space): {env.observation_space}")
    print(f" - Min: {env.observation_space.low}")
    print(f" - Max: {env.observation_space.high}")
    print(" Trạng thái của con lắc bao gồm 3 giá trị liên tục:")
    print(" 1. cos(theta) - theta là góc lệch so với phương thẳng đứng hướng lên (từ -1.0 đến 1.0)")
    print(" 2. sin(theta) - (từ -1.0 đến 1.0)")
    print(" 3. theta_dot  - Vận tốc góc của con lắc (từ -8.0 đến 8.0)")
    
    print(f"\nKhông gian hành động (Action Space): {env.action_space}")
    print(f" - Min: {env.action_space.low}")
    print(f" - Max: {env.action_space.high}")
    print(" Hành động là lực mô-men xoắn (torque) tác động vào trục quay con lắc:")
    print(" - 1 giá trị liên tục trong khoảng [-2.0, 2.0]")
    print("=" * 50)

    num_episodes = 3
    max_steps = 200

    for ep in range(num_episodes):
        state, info = env.reset()
        total_reward = 0
        
        print(f"\n--- Bắt đầu Episode {ep + 1} ---")
        
        for step in range(max_steps):
            # Chọn hành động ngẫu nhiên (nằm trong khoảng [-2.0, 2.0])
            action = env.action_space.sample()
            
            # Thực hiện bước đi trong môi trường
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            state = next_state
            
            # In thông tin của 5 bước đầu để dễ quan sát cấu trúc dữ liệu
            if step < 5:
                print(f"Step {step+1}:")
                print(f"  - Action (Torque): {action[0]:.4f}")
                print(f"  - State (cos, sin, speed): [{state[0]:.4f}, {state[1]:.4f}, {state[2]:.4f}]")
                print(f"  - Reward: {reward:.4f}")
            
            # Tạm dừng một chút để tốc độ render mượt mà, dễ nhìn bằng mắt
            time.sleep(0.02)
            
            if done:
                break
                
        print(f"Kết thúc Episode {ep + 1} | Tổng Reward: {total_reward:.2f}")
        time.sleep(1.0) # Nghỉ 1s giữa các Episode

    env.close()
    print("\nĐã đóng môi trường.")

if __name__ == "__main__":
    main()
