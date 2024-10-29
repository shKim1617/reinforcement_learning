import numpy as np
import torch
import random
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, BaseCallback
from drone_class import Drone, LeaderDrone, FollowerDrone, create_drones
from matplotlib.animation import FuncAnimation
from env_class import DroneFormationEnv
from env_callback import PlottingCallback


# 실행 준비
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = DroneFormationEnv()
env.reset(seed=1)

plotting_callback = PlottingCallback()

callback_list = CallbackList([plotting_callback])

model = PPO(
    "MlpPolicy", 
    env,
    verbose=1,
    device=device,
    n_steps=2048, 
    batch_size=64, 
    learning_rate=1e-4,  # 학습률 스케줄러 적용
    clip_range=0.2,  # 클리핑 범위를 높여 더 유연한 정책 갱신
    gamma=0.995,  # 미래의 보상을 더 중요하게
    gae_lambda=0.98,  # GAE 람다 값 증가
    ent_coef=0.05,  # 탐험적 행동 강화
    use_sde=True,  # 상태 의존 탐색 사용
    sde_sample_freq=4,  # 더 자주 탐험적 행동을 샘플링
    normalize_advantage=True,  # 어드밴티지 정규화
)

loaded_model = PPO.load("one_agent/version_2/test.zip")
obs, _ = env.reset(seed=1)

plt.ioff()
plt.show()

def animate(frame):
    global obs, env
    
    #action = env.action_space.sample()  # 랜덤 액션을 생성
    #observation, reward, terminated, truncated, info = env.step(action)  # 스텝 실행
    action, _ = loaded_model.predict(obs, deterministic=True)
    obs, rewards, terminated, truncated, info = env.step(action)
    
    # 이전 그래픽 지우기
    plt.cla()

    # 드론 위치 다시 그리기
    plt.scatter(env.leader.position[0], env.leader.position[1], color='blue', label='Leader')
    plt.scatter(env.follows[0].position[0], env.follows[0].position[1], color='red', label='Follower 1')
    #plt.scatter(env.follower2.position[0], env.follower2.position[1], color='green', label='Follower 2')

    # 포메이션 라인 그리기
    plt.plot([env.leader.position[0], env.follows[0].position[0]], [env.leader.position[1], env.follows[0].position[1]], 'r--')
    #plt.plot([env.leader.position[0], env.follower2.position[0]], [env.leader.position[1], env.follower2.position[1]], 'g--')

    # 드론들의 방향을 화살표로 그리기 (리더와 팔로워 모두)
    plt.quiver(env.leader.position[0], env.leader.position[1], env.leader.direction[0], env.leader.direction[1], color='blue', scale=5)
    plt.quiver(env.follows[0].position[0], env.follows[0].position[1], env.follows[0].direction[0], env.follows[0].direction[1], color='red', scale=5)
    #plt.quiver(env.follower2.position[0], env.follower2.position[1], env.follower2.orientation[0], env.follower2.orientation[1], color='green', scale=5)

    # 각 드론의 속도 계산 (속도는 벡터 크기)
    leader_speed = np.linalg.norm(env.leader.direction * env.leader_velocity)
    follower1_speed = np.linalg.norm(action)
    #follower2_speed = np.linalg.norm(action[2:4])

    # 드론들의 속도를 텍스트로 표시
    plt.text(env.leader.position[0], env.leader.position[1] + 5, f"Speed: {leader_speed:.2f}", color='blue')
    plt.text(env.follows[0].position[0], env.follows[0].position[1] + 5, f"Speed: {follower1_speed:.2f}", color='red')
    #plt.text(env.follower2.position[0], env.follower2.position[1] + 5, f"Speed: {follower2_speed:.2f}", color='green')

    # 그래프 설정
    # plt.xlim(-200, 200)
    # plt.ylim(-200, 200)
    plt.xlim(env.leader.position[0] - 100, env.leader.position[0] + 100)
    plt.ylim(env.leader.position[1] - 100, env.leader.position[1] + 100)
    plt.title("Drone Formation Movement with Orientation")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.legend()
    plt.grid(True)
    
# 애니메이션 실행
fig = plt.figure(figsize=(6, 6))
ani = FuncAnimation(fig, animate, frames=100, interval=200)
plt.show()