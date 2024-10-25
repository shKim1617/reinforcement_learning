import numpy as np
import torch
import random
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

REL_POS_FOLLOWER1 = [-5, 5]
REL_POS_FOLLOWER2 = [-5, -5]

# 드론 클래스 정의
class Drone:
    def __init__(self, position, is_leader=False):
        self.position = np.array(position, dtype=np.float32)
        self.is_leader = is_leader
        self.orientation = np.array([1, 0, 0], dtype=np.float32)  # 초기 방향 (단위 벡터)

    def update_position(self, velocity):
        self.position += velocity

    def get_relative_distance_and_direction(self, other_drone):
        relative_position = other_drone.position - self.position
        distance = np.linalg.norm(relative_position).astype(np.float32)
        direction = relative_position / distance  # 정규화된 방향 벡터
        relative_direction = np.dot(self._rotation_matrix(), direction).astype(np.float32)
        return distance, relative_direction

    def _rotation_matrix(self):
        angle = np.arctan2(self.orientation[1], self.orientation[0]).astype(np.float32)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        return np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]], dtype=np.float32)

    def set_orientation(self, new_orientation):
        self.orientation = np.array(new_orientation, dtype=np.float32) / np.linalg.norm(new_orientation)


# 환경 정의
class DroneFormationEnv(gym.Env):
    def __init__(self):
        super(DroneFormationEnv, self).__init__()
        self.leader = Drone([0.0, 0.0], is_leader=True)
        self.follower1 = Drone(REL_POS_FOLLOWER1)
        self.follower2 = Drone(REL_POS_FOLLOWER2)
        self.action_space = gym.spaces.Box(low=-2, high=2, shape=(4,), dtype=np.float32)                        # 액션 스페이스
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)         # 서로 관측이 무한대로 가능
        self.max_steps = 500
        self.current_step = 0

    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.leader.position = np.array([0.0, 0.0], dtype=np.float32)
        self.follower1.position = np.array(REL_POS_FOLLOWER1, dtype=np.float32)
        self.follower2.position = np.array(REL_POS_FOLLOWER2, dtype=np.float32)
        self.current_step = 0
        random_angle = random.uniform(-np.pi, np.pi)
        self.leader.orientation = np.array([np.cos(random_angle), np.sin(random_angle)], dtype=np.float32)
        self.leader_velocity = random.uniform(0.5, 5.0)
        
        # Return both the observation and an empty info dict
        return self._get_observation(), {}
        
    def _get_observation(self):
        distance1, direction1 = self.follower1.get_relative_distance_and_direction(self.leader)
        distance2, direction2 = self.follower2.get_relative_distance_and_direction(self.leader)
        return np.concatenate([[distance1], direction1, [distance2], direction2]).astype(np.float32)

    def step(self, action):
        velocity_f1 = action[:2]
        velocity_f2 = action[2:4]
        self.follower1.update_position(velocity_f1)
        self.follower2.update_position(velocity_f2)
        if self.current_step % 100 == 0:
            random_angle = random.uniform(-np.pi, np.pi)
            random_speed = random.uniform(0.5, 2.0)
            self.leader.orientation = np.array([np.cos(random_angle), np.sin(random_angle)], dtype=np.float32)
            self.leader_velocity = random_speed
        leader_velocity = self.leader.orientation * self.leader_velocity
        self.leader.update_position(leader_velocity)
        reward = float(self._compute_reward())
        terminated = self._check_done()
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        return self._get_observation(), reward, terminated, truncated, {}

    def _compute_reward(self):
            if self._check_collision() or self._check_drift():
                return -100
            original_target_follower1 = np.array(REL_POS_FOLLOWER1, dtype=np.float32)
            original_target_follower2 = np.array(REL_POS_FOLLOWER2, dtype=np.float32)
            rotation_matrix = self.leader._rotation_matrix()
            rotated_target_follower1 = np.dot(rotation_matrix, original_target_follower1)
            rotated_target_follower2 = np.dot(rotation_matrix, original_target_follower2)
            global_target_follower1 = self.leader.position + rotated_target_follower1
            global_target_follower2 = self.leader.position + rotated_target_follower2
            distance_follower1 = np.linalg.norm(self.follower1.position - global_target_follower1)
            distance_follower2 = np.linalg.norm(self.follower2.position - global_target_follower2)
            
            
            return - (distance_follower1 + distance_follower2)

    def _check_collision(self):
        distance_between_leader_and_follower1 = np.linalg.norm(self.leader.position - self.follower1.position)
        distance_between_leader_and_follower2 = np.linalg.norm(self.leader.position - self.follower2.position)
        distance_between_followers = np.linalg.norm(self.follower1.position - self.follower2.position)
        if distance_between_followers < 2 or distance_between_leader_and_follower1 < 2 or distance_between_leader_and_follower2 < 2:
            return True
        else:
            return False

    def _check_drift(self):
        distance_between_leader_and_follower1 = np.linalg.norm(self.leader.position - self.follower1.position)
        distance_between_leader_and_follower2 = np.linalg.norm(self.leader.position - self.follower2.position)

        if distance_between_leader_and_follower2 > 10 or distance_between_leader_and_follower1 > 10:
            return True
        else:
            return False


    def _check_done(self):
        return self._check_collision()


env = DroneFormationEnv()

# 환경 리셋 (초기 상태)
observation, info = env.reset()

print("Initial observation:")
print(observation)
print("Initial info:")
print(info)

# 10번의 스텝을 진행하며 결과 출력
for step in range(10):
    print(f"\nStep {step + 1}")
    
    # 랜덤한 액션 생성 (action_space 범위에서)
    action = env.action_space.sample()
    print(f"Action taken: {action}")
    
    # 환경에서 한 스텝 진행
    observation, reward, terminated, truncated, info = env.step(action)
    
    # 결과 출력
    print(f"Observation: {observation}")
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")
    print(f"Info: {info}")
    
    # 에피소드 종료 시 중단
    if terminated or truncated:
        print("Episode ended.")
        break

## 출력결과
# Step 10
# Action taken: [-0.670443   -0.95457196 -0.23839034 -0.01497928]
## 팔로워 드론1,2의 속도 값(x, y)
# Observation: [ 3.0758486   0.11698107 -0.99313414  4.075571    0.2911045   0.95669127]
## 팔로워1과 리더간의 거리 / 팔로워1과 리더 간의 상대 방향 / 
## 팔로워2와 리더 간의 거리 / 팔로워2와 리더 간의 상대 방향
# Reward: -20.51279067993164
# Terminated: False
# Truncated: False
# Info: {}