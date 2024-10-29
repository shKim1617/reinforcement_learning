import numpy as np
import torch
import random
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, BaseCallback

# 수치 정의
REL_POS_FOLLOWER1 = [-20, 20]
REL_POS_FOLLOWER2 = [20, 20]
SAFE_DISTANCE = 5
MAX_DISTANCE = 30
COLLISION_DISTANCE = 1
MAX_STEPS = 1000

# 드론 클래스 정의
class Drone:
    def __init__(self, drone_id, init_position = [0, 0], init_direction = [1, 0]):
        self.drone_id = drone_id
        self.position = np.array(init_position, dtype=np.float32)  # 드론의 위치 (x, y)
        self.direction = np.array(init_direction, dtype=np.float32)
        
    def move(self, target_position):
        self.set_direction(target_position)
        self.set_position(target_position)
                
    def set_position(self, target_position):
        self.position = np.array(target_position, dtype=np.float32)
    
    def set_direction(self, target_position):
        # 목표 위치로의 방향 벡터 계산
        direction_vector = np.array(target_position) - self.position
        # print(direction_vector)
        
        # 방향 벡터의 크기를 계산
        norm = np.linalg.norm(direction_vector)
        
        # 벡터의 크기가 0이 아니면 정규화 (방향 설정), 0이면 방향 유지
        if norm != 0:
            self.direction = direction_vector / norm
    
    def get_state(self):
        return self.drone_id, self.position, self.direction


# 리더 드론
class LeaderDrone(Drone):
    def __init__(self, drone_id=0, init_position=[0, 0], init_direction=[1, 0]):
        # 부모 클래스(Drone)의 초기화 메서드를 호출하여 기본 설정을 상속받음
        super().__init__(drone_id, init_position, init_direction)
        
    # 팔로워 드론을 체크하는 기능
    # 필요한 것인가..?
    def check_follow(self, follow_drone):
        pass

# 팔로워 드론
class FollowerDrone(Drone):
    def __init__(self, drone_id, init_position = [0, 0], init_direction = [1, 0], offset = [0,0]):
        super().__init__(drone_id, init_position, init_direction)
        self.offset = np.array(offset, dtype=np.float32)
        self.leader_position = np.array([0, 0], dtype=np.float32)
        self.distance_to_leader = 15
    
    # 리더 방향에 맞게 오프셋 수정
    # 본인 오프셋을 계속 사용하도록 새로 만들어서 반환
    def calculate_offset(self, leader_direction):
        angle = np.arctan2(leader_direction[1], leader_direction[0])
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        rotation_matrix = np.array(
            [[cos_angle, -sin_angle],
            [sin_angle, cos_angle]],
            dtype=np.float32
        )
        rotated_offset = leader_direction + np.dot(rotation_matrix, self.offset)
        return rotated_offset
    
    def update_leader_position(self, leader_position):
        self.leader_position = np.array(leader_position, dtype=np.float32)
        
    def calculate_distance_to_leader(self, leader_position):
        self.distance_to_leader = np.linalg.norm(leader_position - self.position)

    
    # 여기부터는 사용을 별로 안할 거 같은데.. 강화학습을 이용하려면    

    # 리더 드론을 따라가도록 위치와 방향 설정
    def follow_leader(self, leader_position, leader_direction):
        # 리더 드론의 위치에 오프셋을 적용하여 목표 위치 계산
        target_position = self._calculate_offset(leader_position)
        self.set_direction(target_position)
        self.move(target_position)  # 목표 위치로 이동
        print(self.direction)
        self.direction = leader_direction  # 리더 드론의 방향과 일치하도록 방향 설정
        
    ## 아래의 기능은 gpt가 잘 만들었는데, 사용할 건지는 생각을 해보고..
    
    # 리더 드론과의 거리를 유지하며 편대 유지
    def maintain_distance(self, leader_position, max_distance):
        distance = np.linalg.norm(leader_position - self.position)
        if distance > max_distance:
            self.move_closer(leader_position)  # 거리가 너무 멀면 가까워지도록 이동
    
    # 간단한 충돌 회피 로직 (다른 팔로워와 충돌하지 않도록)
    def avoid_collision(self, other_follower_position):
        distance = np.linalg.norm(other_follower_position - self.position)
        if distance < SAFE_DISTANCE:
            self.move_away(other_follower_position)  # 간단히 충돌 회피를 위한 이동 처리
    
    # 거리가 너무 멀면 리더에게 가까워지는 로직 구현 (추가 기능)
    def move_closer(self, leader_position):
        direction_to_leader = leader_position - self.position
        norm = np.linalg.norm(direction_to_leader)
        if norm != 0:
            direction_to_leader /= norm  # 정규화하여 방향 설정
            self.position += direction_to_leader  # 리더 쪽으로 이동
    
    # 충돌을 피하기 위해 간단히 방향을 반대로 이동 (추가 기능)
    def move_away(self, other_follower_position):
        direction_away = self.position - other_follower_position
        norm = np.linalg.norm(direction_away)
        if norm != 0:
            direction_away /= norm  # 반대 방향으로 정규화
            self.position += direction_away  # 충돌 회피를 위해 반대 방향으로 이동


# 환경 설정
class DroneFormationEnv(gym.Env):
    def __init__(self):
        super(DroneFormationEnv, self).__init__()
        
        # 상태 공간: 리더 위치, 방향 + 팔로워 위치, 방향, 리더와의 거리 = 4 + 5 + 5
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)
        
        # 행동 공간: 팔로워 드론들의 이동 방향 (예: [dx, dy]로 각 드론의 이동을 정의)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        # 리더와 팔로워 드론 초기화
        self.leader_drone = LeaderDrone(drone_id=0)
        self.follower_drone_1 = FollowerDrone(drone_id=1, offset=REL_POS_FOLLOWER1)
        self.follower_drone_2 = FollowerDrone(drone_id=2, offset=REL_POS_FOLLOWER2)
        
        self.leader_velocity = [random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)]
        
        self.max_steps = MAX_STEPS
        self.current_step = 0
        self.terminated = False
        
        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.leader_drone = LeaderDrone(drone_id=0, init_position=[0, 0])
        self.follower_drone_1 = FollowerDrone(drone_id=1, offset=REL_POS_FOLLOWER1)
        self.follower_drone_2 = FollowerDrone(drone_id=2, offset=REL_POS_FOLLOWER2)
        self.leader_velocity = [random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)]
        self.current_step = 0
        self.terminated = False
        
        # 초기 상태 반환 및 추가 정보 반환
        obs = self._get_obs()

        # obs가 숫자형 배열인지 확인
        try:
            obs = np.array(obs, dtype=np.float32)  # obs를 숫자형 배열로 변환
        except ValueError as e:
            print(f"Error converting obs to numeric array: {e}")

        # 초기 상태 반환
        return self._get_obs(), {}
    
    def _get_obs(self):
        leader_state = np.concatenate((self.leader_drone.position, self.leader_drone.direction))
        
        # # np.array[] 대신 np.array()로 수정
        # follower_state_1 = np.concatenate((self.follower_drone_1.position, self.follower_drone_1.direction, 
        #                                 np.array([self.follower_drone_1.distance_to_leader])))
        
        # follower_state_2 = np.concatenate((self.follower_drone_2.position, self.follower_drone_2.direction, 
        #                                 np.array([self.follower_drone_2.distance_to_leader])))
        
        # # 상태에 NaN 값이 포함되는지 체크
        # obs = np.concatenate((leader_state, follower_state_1, follower_state_2))
        # if np.isnan(obs).any():
        #     print("NaN detected in observation!")

        # # 리더와 팔로워 2대의 상태를 하나로 합쳐 반환 (14차원)
        # return np.concatenate((leader_state, follower_state_1, follower_state_2))
        # 팔로워 드론 1과 2의 상태를 숫자형 배열로 결합

        try:
            leader_state = np.concatenate((self.leader_drone.position, self.leader_drone.direction))
            if np.isnan(leader_state).any():
                print("NaN detected in leader_state!")
            
            follower_state_1 = np.concatenate((self.follower_drone_1.position, self.follower_drone_1.direction, 
                                            np.array([self.follower_drone_1.distance_to_leader], dtype=np.float32)))
            if np.isnan(follower_state_1).any():
                print("NaN detected in follower_state_1!")

            follower_state_2 = np.concatenate((self.follower_drone_2.position, self.follower_drone_2.direction, 
                                            np.array([self.follower_drone_2.distance_to_leader], dtype=np.float32)))
            if np.isnan(follower_state_2).any():
                print("NaN detected in follower_state_2!")

            obs = np.concatenate((leader_state, follower_state_1, follower_state_2))
            if np.isnan(obs).any():
                print("NaN detected in observation!")
        
        except Exception as e:
            print(f"Error in _get_obs: {e}")
            obs = None

        return obs


    def step(self, action):
        # 리더 드론 이동   
        # 리더 드론의 위치를 무작위로 설정 (-1.0에서 1.0 사이의 무작위 값)
        if self.current_step % 100 == 0:
            self.leader_velocity = [random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)]
            
        # 리더 드론을 속도에 따라 위치 업데이트
        new_position = np.array(self.leader_drone.position) + np.array(self.leader_velocity)
        self.leader_drone.move(new_position)
        
        # 팔로워 드론들은 행동(action)에 따라 이동
        self.follower_drone_1.move(action[:2])
        self.follower_drone_2.move(action[2:])
        
        # 팔로워 드론들이 리더와의 거리를 계산
        self.follower_drone_1.calculate_distance_to_leader(self.leader_drone.position)
        self.follower_drone_2.calculate_distance_to_leader(self.leader_drone.position)
        
        self.current_step += 1
        
        # 현재 상태 반환
        obs = self._get_obs()
                
        # 보상 계산, 종료 조건 확인 등 추가 로직 포함 가능
        reward = self._calculate_reward()
        # 종료 조건 확인
        terminated = self._is_done()
        
        # truncated (예: 시간이 초과되었을 때 종료되는지 여부)
        truncated = self.current_step >= self.max_steps                                       ## 여기 수정 필요
        
        return obs, reward, terminated, truncated, {}

    def _calculate_reward(self):
        reward = 0
        
        # 거리 기반 보상
        reward += self._calculate_distance_offset()
        
        # 세이프 거리 내에서 패널티
        reward += self._calculate_safe_distance()
        
        # 충돌 감지 패널티
        reward += self._check_collision()
        
        # 너무 멀리 있는지 확인 패널티
        reward += self._check_too_far()
        
        # 보상에 NaN 값이 있는지 확인
        if np.isnan(reward):
            print("NaN detected in reward calculation!")

        return reward
    
    def _calculate_distance_offset(self):
        reward = 0
        
        # 팔로워 드론 1과 2의 목표 오프셋 위치 계산
        target_position_1 = self.follower_drone_1.calculate_offset(self.leader_drone.direction) + self.leader_drone.position
        target_position_2 = self.follower_drone_2.calculate_offset(self.leader_drone.direction) + self.leader_drone.position

        # 실제 팔로워 드론 1과 2의 위치와 목표 위치 간의 거리 계산
        dist_follower_1 = np.linalg.norm(self.follower_drone_1.position - target_position_1)
        dist_follower_2 = np.linalg.norm(self.follower_drone_2.position - target_position_2)
        
        # 팔로워 드론이 목표 위치에서 ±5 범위 안에 있으면 보상 부여
        reward += max(0, 1 - dist_follower_1 / 30.0)  # 30은 최대 거리 기준
        reward += max(0, 1 - dist_follower_2 / 30.0)

        return reward
    
    def _calculate_safe_distance(self):
        reward = 0
            
        if self.follower_drone_1.distance_to_leader < SAFE_DISTANCE:
            reward -= 1
        if self.follower_drone_2.distance_to_leader < SAFE_DISTANCE:
            reward -= 1

        return reward
    
    def _check_collision(self):
        reward = 0
        
        # 팔로워 드론들 간 충돌
        distance_between_followers = np.linalg.norm(self.follower_drone_1.position - self.follower_drone_2.position)
        
        if distance_between_followers < COLLISION_DISTANCE:
            reward -= 10  # 충돌 시 패널티 부여
            
        # 리더 드론과 충돌
        if self.follower_drone_1.distance_to_leader < COLLISION_DISTANCE:
            reward -= 10  # 너무 멀어지면 패널티 부여
        if self.follower_drone_2.distance_to_leader < COLLISION_DISTANCE:
            reward -= 10

        return reward

    def _check_too_far(self):
        reward = 0

        # 팔로워 드론 1과 2가 리더 드론으로부터 너무 멀어졌는지 확인
        if self.follower_drone_1.distance_to_leader > MAX_DISTANCE:
            reward -= 1  # 너무 멀어지면 패널티 부여
        if self.follower_drone_2.distance_to_leader > MAX_DISTANCE:
            reward -= 1

        return reward
    
    def _is_done(self):
        """
        종료 조건을 정의하는 메소드.
        보통 학습 환경에서 특정 조건(예: 타임스텝 초과, 실패 등)을 기반으로 종료.
        """
        # 간단한 예시: 리더와 팔로워가 너무 멀어지면 종료
        dist_follower_1 = np.linalg.norm(self.leader_drone.position - self.follower_drone_1.position)
        dist_follower_2 = np.linalg.norm(self.leader_drone.position - self.follower_drone_2.position)
        
        if dist_follower_1 > 100 or dist_follower_2 > 100 or self.current_step > MAX_STEPS:
            return True  # 종료 조건 만족
        
        return False
    
# # matplotlib 설정
is_ipython = 'inline' in plt.get_backend()
if is_ipython:
    from IPython import display

plt.ion()


def plot_durations(episode_durations, show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # 100개의 에피소드 평균을 가져 와서 도표 그리기
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # 도표가 업데이트되도록 잠시 멈춤
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
            
class PlottingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(PlottingCallback, self).__init__(verbose)
        self.episode_durations = []

    def _on_step(self) -> bool:
        # 매 스텝마다 호출되는 함수
        # 'dones' 배열에서 현재 환경의 종료 상태를 확인
        if self.locals['dones'][0]:
            # 에피소드가 종료되면 episode length를 기록
            episode_duration = self.locals['infos'][0].get('episode', {}).get('l', 0)
            self.episode_durations.append(episode_duration)
            plot_durations(self.episode_durations)  # 클래스 내부의 episode_durations를 전달

        return True  # 계속 학습하기 위해서는 True 반환

    def _on_training_end(self) -> None:
        # 학습이 끝났을 때 최종 결과를 표시
        plot_durations(self.episode_durations, show_result=True)
    
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

model.learn(total_timesteps=500000, callback=callback_list)

model.save("multi_agent_v.1/test")

print('Complete')
plot_durations(plotting_callback.episode_durations, show_result=True)
plt.ioff()
plt.show()


# # 이어서 학습
# loaded_model_name = "multi_agent_v.1/test.zip"
# loaded_model = PPO.load(
#     loaded_model_name, 
#     env=env, 
#     device=device)

# loaded_model.learn(total_timesteps=950000, callback=callback_list)

# save_model_name = "multi_agent_v.1/test100"
# loaded_model.save(save_model_name)

# # 여기에 무슨 애니메이션을 만들거냐면 학습이 어떻게 이루어지고 있는지 볼거여
# print('Complete')
# plot_durations(plotting_callback.episode_durations, show_result=True)
# plt.ioff()
# plt.show()