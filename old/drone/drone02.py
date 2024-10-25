import numpy as np
import gym
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 드론 클래스 정의
class Drone:
    def __init__(self, position, is_leader=False):
        self.position = np.array(position)
        self.is_leader = is_leader
        self.orientation = np.array([0, 1, 0])  # 초기 방향 (단위 벡터)

    def update_position(self, velocity):
        # 속도를 바탕으로 위치 업데이트
        self.position += velocity

    def get_relative_distance_and_direction(self, other_drone):
        # 다른 드론과의 상대적 거리와 방향 계산 (자신의 좌표계를 기준으로)
        relative_position = other_drone.position - self.position
        distance = np.linalg.norm(relative_position)

        # 방향 계산 (현재 드론의 좌표계를 기준으로)
        direction = relative_position / distance  # 정규화된 방향 벡터

        # 자신의 좌표계에서의 상대 방향을 얻기 위해 방향을 회전시킴
        relative_direction = np.dot(self._rotation_matrix(), direction)
        return distance, relative_direction

    def _rotation_matrix(self):
        # 2D 드론이라 가정하고, orientation을 기반으로 회전 행렬 생성
        angle = np.arctan2(self.orientation[1], self.orientation[0])
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
        return rotation_matrix

    def set_orientation(self, new_orientation):
        # 드론의 방향 업데이트
        self.orientation = np.array(new_orientation) / np.linalg.norm(new_orientation)


# 환경 정의
class DroneFormationEnv(gym.Env):
    def __init__(self):
        super(DroneFormationEnv, self).__init__()

        # Leader와 2개의 follower 드론 생성
        self.leader = Drone([0.0, 0.0], is_leader=True)
        self.follower1 = Drone([10.0, 10.0])
        self.follower2 = Drone([10.0, -10.0])

        # Action space: follower 드론들의 움직임 (2D)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,))  # x, y for two followers

        # Observation space: follower 드론에서 보는 리더의 상대적 거리 및 방향
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,))  # 2 followers

    def reset(self):
        # 드론 위치 초기화
        self.leader.position = np.array([0.0, 0.0])
        self.follower1.position = np.array([10.0, 10.0])
        self.follower2.position = np.array([10.0, -10.0])

        return self._get_observation()

    def _get_observation(self):
        # 각 follower가 리더 드론과의 거리 및 방향 정보를 얻음
        distance1, direction1 = self.follower1.get_relative_distance_and_direction(self.leader)
        distance2, direction2 = self.follower2.get_relative_distance_and_direction(self.leader)

        # 상태는 follower1과 follower2의 리더와의 거리 및 방향
        return np.concatenate([[distance1], direction1, [distance2], direction2])

    def step(self, action):
        # action은 follower 드론들의 이동 방향을 나타냄 (follower1의 x, y, follower2의 x, y)
        velocity_f1 = action[:2]
        velocity_f2 = action[2:]

        # 팔로워 드론의 위치 업데이트
        self.follower1.update_position(velocity_f1)
        self.follower2.update_position(velocity_f2)

        # 리더 드론은 앞으로 이동 (간단히 가정하여 +x 방향으로 이동)
        self.leader.update_position(np.array([1.0, 0.0]))  # 리더는 고정된 방향으로 이동한다고 가정

        # 보상 계산
        reward = self._compute_reward()

        # 에피소드 종료 조건: 충돌 여부
        done = self._check_done()

        return self._get_observation(), reward, done, {}

    def _compute_reward(self):
        # 팔로워 드론이 리더와 대형을 유지하면 보상, 벗어나면 페널티
        distance1, _ = self.follower1.get_relative_distance_and_direction(self.leader)
        distance2, _ = self.follower2.get_relative_distance_and_direction(self.leader)

        reward = - (distance1 + distance2)  # 거리 기반 페널티

        # 충돌하면 큰 패널티
        if self._check_collision():
            reward -= 100

        return reward

    def _check_collision(self):
        # 두 팔로워 드론 간의 충돌 여부를 체크
        distance_between_followers = np.linalg.norm(self.follower1.position - self.follower2.position)
        return distance_between_followers < 2  # 임계값

    def _check_done(self):
        # 일정 시간 이상 경과하거나 충돌하면 에피소드 종료
        return self._check_collision()

# 환경 생성
env = DroneFormationEnv()

# GUI용 matplotlib 설정
fig, ax = plt.subplots()
ax.set_xlim(-20, 40)
ax.set_ylim(-20, 20)

# 여기서 여러개의 값을 투플로 받는데
# 맨 앞의 값 하나만 받으려고 콤마 넣어놨군
# 그 이후의 값을 버린다는 거네
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
def update(frame):
    # 현재 무작위 값을 때려넣으면서 학습만 시키고 있음
    # 이것이 팔로워 드론의 값을 유의미하게 조정하는 것이 아님
    # 리더 드론을 따라가는 로직이 빠져있음
    action = env.action_space.sample()  # 무작위 액션 (학습 전에 랜덤 움직임)
    obs, reward, done, info = env.step(action)
    
    # 드론들의 위치 업데이트
    leader_point.set_data([env.leader.position[0]], [env.leader.position[1]])
    follower1_point.set_data([env.follower1.position[0]], [env.follower1.position[1]])
    follower2_point.set_data([env.follower2.position[0]], [env.follower2.position[1]])

    return leader_point, follower1_point, follower2_point


# 애니메이션 실행
ani = FuncAnimation(fig, update, frames=np.arange(0, 100), init_func=init, blit=True, repeat=False)

# 시각화
plt.legend()
plt.show()


