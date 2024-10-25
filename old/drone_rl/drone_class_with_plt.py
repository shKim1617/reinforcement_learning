# 드론 클래스를 이용하여 그림으로 나타내기
# 아직 환경 설정과 학습을 하지 않음
# 일단은 리더의 움직임을 만들어 놓고
# 팔로우 드론들이 따라가는 모습을 나타내기만 함. 방향 포함

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


leader_drone = Drone((0, 0), is_leader=True)
follower1_drone = Drone([-5, 5])
follower2_drone = Drone([-5, -5])

# 위치 업데이트 함수
def update_drones():
    # 리더 드론은 임의로 이동 (예: 속도 벡터 [0.1, 0.1] 방향으로)
    leader_velocity = np.array([0.1, 0.1])
    leader_drone.update_position(leader_velocity)
    leader_drone.set_orientation(leader_velocity)  # 이동 방향으로 방향 설정
    
    # 팔로워 드론들은 리더 드론을 향해 일정한 속도로 따라감
    follower1_velocity = (leader_drone.position - follower1_drone.position) * 0.05
    follower2_velocity = (leader_drone.position - follower2_drone.position) * 0.05
    follower1_drone.update_position(follower1_velocity)
    follower2_drone.update_position(follower2_velocity)
    
    # 팔로워 드론들도 이동 방향에 따라 방향을 업데이트
    follower1_drone.set_orientation(follower1_velocity)
    follower2_drone.set_orientation(follower2_velocity)

def animate(frame):
    # 드론들의 위치 업데이트
    update_drones()
    
    # 이전 그래픽 지우기
    plt.cla()
    
    # 드론 위치 다시 그리기
    plt.scatter(leader_drone.position[0], leader_drone.position[1], color='blue', label='Leader')
    plt.scatter(follower1_drone.position[0], follower1_drone.position[1], color='red', label='Follower 1')
    plt.scatter(follower2_drone.position[0], follower2_drone.position[1], color='green', label='Follower 2')

    # 포메이션 라인 그리기
    plt.plot([leader_drone.position[0], follower1_drone.position[0]], 
             [leader_drone.position[1], follower1_drone.position[1]], 'r--')
    plt.plot([leader_drone.position[0], follower2_drone.position[0]], 
             [leader_drone.position[1], follower2_drone.position[1]], 'g--')
    
    # 드론들의 방향을 화살표로 그리기 (리더와 팔로워 모두)
    plt.quiver(leader_drone.position[0], leader_drone.position[1], leader_drone.orientation[0], leader_drone.orientation[1], color='blue', scale=10)
    plt.quiver(follower1_drone.position[0], follower1_drone.position[1], follower1_drone.orientation[0], follower1_drone.orientation[1], color='red', scale=10)
    plt.quiver(follower2_drone.position[0], follower2_drone.position[1], follower2_drone.orientation[0], follower2_drone.orientation[1], color='green', scale=10)

    # 그래프 설정
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.title("Drone Formation Movement with Orientation")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.legend()
    plt.grid(True)

# 애니메이션 실행
fig = plt.figure(figsize=(6,6))
ani = FuncAnimation(fig, animate, frames=100, interval=200)

plt.show()