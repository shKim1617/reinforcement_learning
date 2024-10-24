import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

import time

# 드론 클래스 정의
class Drone:
    def __init__(self, position, is_leader=False):
        self.position = np.array(position, dtype=np.float32)
        self.is_leader = is_leader
        self.orientation = np.array([0, 1, 0], dtype=np.float32)  # 초기 방향 (단위 벡터)

    def update_position(self, velocity):
        # 속도를 바탕으로 위치 업데이트
        self.position += velocity

    def get_relative_distance_and_direction(self, other_drone):
        # 다른 드론과의 상대적 거리와 방향 계산 (자신의 좌표계를 기준으로)
        relative_position = other_drone.position - self.position
        distance = np.linalg.norm(relative_position).astype(np.float32)

        # 방향 계산 (현재 드론의 좌표계를 기준으로)
        direction = relative_position / distance  # 정규화된 방향 벡터

        # 자신의 좌표계에서의 상대 방향을 얻기 위해 방향을 회전시킴
        relative_direction = np.dot(self._rotation_matrix(), direction).astype(np.float32)
        return distance, relative_direction

    def _rotation_matrix(self):
        # 2D 드론이라 가정하고, orientation을 기반으로 회전 행렬 생성
        angle = np.arctan2(self.orientation[1], self.orientation[0]).astype(np.float32)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]], dtype=np.float32)
        return rotation_matrix

    def set_orientation(self, new_orientation):
        # 드론의 방향 업데이트
        self.orientation = np.array(new_orientation, dtype=np.float32) / np.linalg.norm(new_orientation)


# 환경 정의
class DroneFormationEnv(gym.Env):
    def __init__(self):
        super(DroneFormationEnv, self).__init__()

        # Leader와 2개의 follower 드론 생성
        self.leader = Drone([0.0, 0.0], is_leader=True)
        self.follower1 = Drone([10.0, 10.0])
        self.follower2 = Drone([10.0, -10.0])

        # Action space: follower 드론들의 이동 (2D)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)  # x, y for two followers

        # Observation space: follower 드론에서 보는 리더의 상대적 거리 및 방향 (6차원: 2개의 드론 각각 거리(1) + 방향(2))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self.seed_val = None  # seed를 저장할 변수
        self.max_steps = 200  # 최대 스텝 수                                            ## 초기값 100
        self.current_step = 0  # 현재 스텝

    def reset(self, seed=None, options=None):
        # seed 초기화
        if seed is not None:
            self.seed_val = seed
            np.random.seed(self.seed_val)

        # 드론 위치 초기화
        self.leader.position = np.array([0.0, 0.0], dtype=np.float32)
        self.follower1.position = np.array([10.0, 10.0], dtype=np.float32)
        self.follower2.position = np.array([10.0, -10.0], dtype=np.float32)

        self.current_step = 0  # 스텝 초기화
        
        # debug
        #print(f"범인은 너냐")

        return self._get_observation(), {}

    def _get_observation(self):
        # 각 follower가 리더 드론과의 거리 및 방향 정보를 얻음
        distance1, direction1 = self.follower1.get_relative_distance_and_direction(self.leader)
        distance2, direction2 = self.follower2.get_relative_distance_and_direction(self.leader)
        
        # 서로 간의 거리도 구해야 함
        # 근데 이걸 여기서 사용할 이유가 있나
        # distance3, direction3 = self.follower1.get_relative_distance_and_direction(self.follower2)

        # 상태는 follower1과 follower2의 리더와의 거리 및 방향
        # 각 팔로워 드론의 거리와 방향을 결합하여 하나의 상태 벡터로 반환
        return np.concatenate([[distance1], direction1, [distance2], direction2]).astype(np.float32)

    def step(self, action):
        # action은 follower 드론들의 이동 방향을 나타냄 (follower1의 x, y, follower2의 x, y)
        velocity_f1 = action[:2]
        velocity_f2 = action[2:]

        # 팔로워 드론의 위치 업데이트
        self.follower1.update_position(velocity_f1)
        self.follower2.update_position(velocity_f2)

        ## 디버깅 코드
        # print(f"leader position: {self.leader.position}")
        
        # 리더 드론은 앞으로 이동 (간단히 가정하여 +x 방향으로 이동)
        self.leader.update_position(np.array([1.0, 0.0], dtype=np.float32))  # 리더는 고정된 방향으로 이동한다고 가정               # 초기값 1, 0
        
        
        # print(f"leader position: {self.leader.position}")
        # print()
        #time.sleep(1)
        

        # 보상 계산
        reward = float(self._compute_reward())  # 보상을 float로 변환

        # 에피소드 종료 조건: 충돌 여부
        terminated = self._check_done()

        # 최대 스텝 수에 도달했는지 확인 (truncated 처리)
        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        return self._get_observation(), reward, terminated, truncated, {}

    def _compute_reward(self):
        # 팔로워 드론이 리더와 대형을 유지하면 보상, 벗어나면 페널티
        distance1, _ = self.follower1.get_relative_distance_and_direction(self.leader)
        distance2, _ = self.follower2.get_relative_distance_and_direction(self.leader)
        
        # 서로 간의 거리
        distance3, _ = self.follower1.get_relative_distance_and_direction(self.follower2) 

        # 이거 완전히 바꿔야 함
        # reward = - (distance1 + distance2)  # 거리 기반 페널티
        
        reward = 0
        base_distance = 14.14
        follower_distance = 20
        
        diff_distance1 = abs(distance1 - base_distance)
        diff_distance2 = abs(distance2 - base_distance)
        diff_distance3 = abs(distance3 - follower_distance)
        
        # if diff_distance1 < 5:
        #     reward += 20 - diff_distance1
        # else:
        #     reward -= diff_distance1
        
        # if diff_distance2 < 5:
        #     reward += 20 - diff_distance2
        # else:
        #     reward -= diff_distance2
        
        # if diff_distance3 < 5:
        #     reward += 5 - diff_distance3
        # else:
        #     reward -= diff_distance3 - 5
            
        # 보상은 가능한한 음수로 주는 게 좋대
        # 양수의 보상은 보상을 얻기위하여 이상한 짓을 한다는군
        # 겁나 똑똑한데
        reward = -(diff_distance1 + diff_distance2 + diff_distance3)
        
        

        # 충돌하면 큰 페널티
        # 그래서 환경을 리셋시키게 되는 거고
        if self._check_collision():
            reward -= 100

        return reward

    def _check_collision(self):
        # 두 팔로워 드론 간의 충돌 여부를 체크
        distance_between_followers = np.linalg.norm(self.follower1.position - self.follower2.position)
        if distance_between_followers > 2:  # 임계값
            return False
        else:
            return True

    def _check_done(self):
        # 일정 시간 이상 경과하거나 충돌하면 에피소드 종료
        return self._check_collision()

# 모델 로드
model_path = "drone/ppo_drone_formation_model00.zip"
model = PPO.load(model_path)

# 환경 초기화 (이 부분은 사용자가 사용하는 환경으로 대체 필요)
# 예시: env = gym.make('CartPole-v1')
env = DroneFormationEnv()
  # 여기에 환경을 설정하세요

# 환경이 올바르게 정의되었는지 확인
check_env(env)

# GUI용 matplotlib 설정
fig, ax = plt.subplots()
ax.set_xlim(-20, 200)
ax.set_ylim(-20, 20)

leader_point, = ax.plot([], [], 'ro', label='Leader')  # 리더 드론은 빨간색
follower1_point, = ax.plot([], [], 'bo', label='Follower 1')  # 팔로워 1 드론은 파란색
follower2_point, = ax.plot([], [], 'go', label='Follower 2')  # 팔로워 2 드론은 초록색

# 초기화 함수
def init():
    leader_point.set_data([], [])
    follower1_point.set_data([], [])
    follower2_point.set_data([], [])
    return leader_point, follower1_point, follower2_point

# 애니메이션 업데이트 함수
obs, _ = env.reset()  # 환경을 초기화하고 obs를 한 번 설정

def update(frame):
    global obs
    # PPO 모델로부터 액션 예측
    action, _states = model.predict(obs, deterministic=True)
    
    # step을 통해 드론의 상태 업데이트
    obs, reward, terminated, truncated, _ = env.step(action)
    
    # 드론들의 위치 업데이트
    leader_point.set_data([env.leader.position[0]], [env.leader.position[1]])
    follower1_point.set_data([env.follower1.position[0]], [env.follower1.position[1]])
    follower2_point.set_data([env.follower2.position[0]], [env.follower2.position[1]])


    if terminated or truncated:
        obs, _ = env.reset()  # 에피소드 종료 시 환경 재설정

    return leader_point, follower1_point, follower2_point

# 애니메이션 실행
ani = FuncAnimation(fig, update, frames=np.arange(0, 100), init_func=init, blit=True, repeat=False)

# 시각화
plt.legend()
plt.show()
