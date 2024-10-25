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
        ## action print
        print("action:", action)
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
    
def animate(frame):
    global env
    action = env.action_space.sample()  # 랜덤 액션을 생성
    observation, reward, terminated, truncated, info = env.step(action)  # 스텝 실행
    
    # 이전 그래픽 지우기
    plt.cla()

    # 드론 위치 다시 그리기
    plt.scatter(env.leader.position[0], env.leader.position[1], color='blue', label='Leader')
    plt.scatter(env.follower1.position[0], env.follower1.position[1], color='red', label='Follower 1')
    plt.scatter(env.follower2.position[0], env.follower2.position[1], color='green', label='Follower 2')

    # 포메이션 라인 그리기
    plt.plot([env.leader.position[0], env.follower1.position[0]], [env.leader.position[1], env.follower1.position[1]], 'r--')
    plt.plot([env.leader.position[0], env.follower2.position[0]], [env.leader.position[1], env.follower2.position[1]], 'g--')

    # 드론들의 방향을 화살표로 그리기 (리더와 팔로워 모두)
    plt.quiver(env.leader.position[0], env.leader.position[1], env.leader.orientation[0], env.leader.orientation[1], color='blue', scale=5)
    plt.quiver(env.follower1.position[0], env.follower1.position[1], env.follower1.orientation[0], env.follower1.orientation[1], color='red', scale=5)
    plt.quiver(env.follower2.position[0], env.follower2.position[1], env.follower2.orientation[0], env.follower2.orientation[1], color='green', scale=5)

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


# 환경 초기화
env = DroneFormationEnv()
env.reset()

# 애니메이션 실행
fig = plt.figure(figsize=(6, 6))
ani = FuncAnimation(fig, animate, frames=100, interval=200)
plt.show()

# 이 환경을 실행만 시킨 것이기 때문에 학습이 이루어지지 않아
# 팔로워 드론이 거의 제자리에서 움찔거리기만 한 모습이 포착
# 급격한 값의 변화를 겪으며 아예 움직임이 제한되는 상황이 발생
# 이걸 ppo 학습을 통해서 어떻게 풀어나갈지가 아주 중요
