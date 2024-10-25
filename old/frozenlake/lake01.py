import gym
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from matplotlib import colors

# OpenAI Gym 환경인 FrozenLake 불러오기
env = gym.make("FrozenLake-v1", is_slippery=False)

# 환경의 상태 및 행동의 수
state_size = env.observation_space.n  # 상태의 개수
action_size = env.action_space.n      # 행동의 개수

# Q 테이블 초기화 (state_size x action_size 크기의 테이블)
Q_table = np.zeros((state_size, action_size))

# 하이퍼파라미터 설정
learning_rate = 0.8     # 학습률 (α)
discount_rate = 0.95    # 할인율 (γ)
exploration_rate = 1.0  # 탐험 비율 (ε)
max_exploration_rate = 1.0
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

# 에피소드 수
num_episodes = 1000
max_steps_per_episode = 100  # 각 에피소드 당 최대 단계 수

# 보상 저장 리스트
rewards_all_episodes = []

# 색상 설정
cmap = colors.ListedColormap(['lightgray', 'blue', 'green', 'red'])
bounds = [0, 1, 2, 3, 4]
norm = colors.BoundaryNorm(bounds, cmap.N)

def plot_frozenlake(state):
    grid = np.zeros((4, 4))
    grid[state // 4, state % 4] = 1
    plt.imshow(grid, cmap=cmap, norm=norm)
    plt.grid(True)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.draw()
    plt.pause(0.1)
    plt.clf()

# Q-learning 알고리즘 실행
for episode in range(num_episodes):
    # 시작 상태 초기화
    state = env.reset()
    done = False
    rewards_current_episode = 0

    for step in range(max_steps_per_episode):
        # 탐험-활용 트레이드오프 결정 (ε-greedy)
        exploration_threshold = random.uniform(0, 1)
        if exploration_threshold > exploration_rate:
            action = np.argmax(Q_table[state, :])  # 최적의 행동 선택
        else:
            action = env.action_space.sample()  # 무작위 행동 선택

        # 행동을 실행하고 환경으로부터 새로운 상태와 보상 받기
        new_state, reward, done, _ = env.step(action)

        # Q 테이블 업데이트 (Q-learning 공식 사용)
        Q_table[state, action] = Q_table[state, action] + learning_rate * (reward + discount_rate * np.max(Q_table[new_state, :]) - Q_table[state, action])

        # 상태 업데이트
        state = new_state

        # 환경 시각화 (매 행동 후 출력)
        plot_frozenlake(state)

        if done:
            break

    # 탐험 비율 감소 (에피소드마다 탐험 비율을 조금씩 줄여나감)
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)

    # 에피소드 별 보상 저장
    rewards_all_episodes.append(rewards_current_episode)

env.close()
