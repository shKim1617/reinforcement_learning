import numpy as np
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from matplotlib.animation import FuncAnimation
from env_class import DroneFormationEnv
from config import LOAD_BEST_MODEL_PATH, LOAD_MODEL_PATH


# 실행 준비
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = DroneFormationEnv()
#env.reset(seed=1)

# 이것도 내가 사용중인 모델이랑 다른데
# 모델을 따로 정리해놔야하나
# model = PPO(
#     "MlpPolicy", 
#     env,
#     verbose=1,
#     device=device,
#     n_steps=2048, 
#     batch_size=64, 
#     learning_rate=1e-4,  # 학습률 스케줄러 적용
#     clip_range=0.2,  # 클리핑 범위를 높여 더 유연한 정책 갱신
#     gamma=0.995,  # 미래의 보상을 더 중요하게
#     gae_lambda=0.98,  # GAE 람다 값 증가
#     ent_coef=0.05,  # 탐험적 행동 강화
#     use_sde=True,  # 상태 의존 탐색 사용
#     sde_sample_freq=4,  # 더 자주 탐험적 행동을 샘플링
#     normalize_advantage=True,  # 어드밴티지 정규화
# )

model = PPO("MlpPolicy", env, verbose=1, device=device)

loaded_model_pass = LOAD_MODEL_PATH
loaded_best_model_pass = LOAD_BEST_MODEL_PATH
loaded_model = PPO.load(loaded_best_model_pass)                                                 # 두 가지 모델 중에 결정
obs, _ = env.reset(seed=1)

plt.ioff()
plt.show()

def animate(frame):
    global obs, env

    action, _ = loaded_model.predict(obs, deterministic=True)
    obs, rewards, terminated, truncated, info = env.step(action)
    
    # 이전 그래픽 지우기
    plt.cla()

    # 드론 위치 다시 그리기
    # 리더
    plt.scatter(env.leader.position[0], env.leader.position[1], color='red', label='Leader')
    # 팔로워1
    plt.scatter(env.follows[0].position[0], env.follows[0].position[1], color='blue', label='Follower 1')
    plt.scatter(env.follows[0].target_position[0], env.follows[0].target_position[1], color='black', label='target_position1')
    # 팔로워2
    plt.scatter(env.follows[1].position[0], env.follows[1].position[1], color='green', label='Follower 2')
    plt.scatter(env.follows[1].target_position[0], env.follows[1].target_position[1], color='black', label='target_position2')

    # 포메이션 라인 그리기
    plt.plot([env.leader.position[0], env.follows[0].position[0]], [env.leader.position[1], env.follows[0].position[1]], 'b--')
    plt.plot([env.leader.position[0], env.follows[1].position[0]], [env.leader.position[1], env.follows[1].position[1]], 'g--')

    # 리더 방향 화살표로 그리기
    plt.quiver(env.leader.position[0], env.leader.position[1], env.leader.direction[0], env.leader.direction[1], color='red', scale=10)

    # 각 드론의 속도 계산 (속도는 벡터 크기)
    leader_speed = np.linalg.norm(env.leader.direction * env.leader_velocity)
    follower1_speed = np.linalg.norm(action[:2])
    follower2_speed = np.linalg.norm(action[2:])

    # 드론들의 속도를 텍스트로 표시
    plt.text(env.leader.position[0], env.leader.position[1] + 1, f"Speed: {leader_speed:.2f}", color='red')
    plt.text(env.follows[0].position[0], env.follows[0].position[1] + 1, f"Speed: {follower1_speed:.2f}", color='blue')
    plt.text(env.follows[1].position[0], env.follows[1].position[1] + 1, f"Speed: {follower2_speed:.2f}", color='green')
    
    # 충돌
    if terminated:
        print("충돌발생!!!!!!!!!!!!")

    # 그래프 설정
    preset = 100
    plt.xlim(env.leader.position[0] - preset, env.leader.position[0] + preset)
    plt.ylim(env.leader.position[1] - preset, env.leader.position[1] + preset)
    plt.title("Drone Formation Movement with Orientation")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.legend()
    plt.grid(True)
    
# 애니메이션 실행
fig = plt.figure(figsize=(6, 6))
ani = FuncAnimation(fig, animate, frames=100, interval=200)
plt.show()