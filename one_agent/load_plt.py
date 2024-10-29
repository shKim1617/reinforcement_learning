import numpy as np
import torch
import random
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, BaseCallback
from drone_class import Drone, LeaderDrone, FollowerDrone, create_drones
from matplotlib.animation import FuncAnimation

# 수치 정의
NUM_FOLLOWS = 1
REL_POS_FOLLOWERS = [[-20, 20], [20, 20]]

SAFE_DISTANCE = 5
MAX_DISTANCE = 30
COLLISION_DISTANCE = 1
MAX_STEPS = 1000

# 환경 설정
class DroneFormationEnv(gym.Env):
    def __init__(self):
        super(DroneFormationEnv, self).__init__()
        
        # 상태 공간: 리더 위치, 방향 + 팔로워 위치, 방향, 리더와의 거리 = 4 + 5
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        
        # 행동 공간: 팔로워 드론들의 이동 방향 (예: [dx, dy]로 각 드론의 이동을 정의)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        
        # 리더와 팔로워 드론 초기화
        self.leader, self.follows = create_drones(NUM_FOLLOWS)
        
        self.leader_velocity = 0
        
        self.max_steps = MAX_STEPS
        self.current_step = 0
        self.terminated = False
        
        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.leader, self.follows = create_drones(NUM_FOLLOWS)
        self.leader_velocity = 0
        self.current_step = 0
        self.terminated = False
        
        # 초기 상태 반환 및 추가 정보 반환
        obs = self._get_obs()

        return obs, {}
    
    def _get_obs(self):
        leader_state = np.concatenate((self.leader.position, self.leader.direction))
        
        follows_state = []
        for i in range(NUM_FOLLOWS):
            follows_state.append(np.concatenate((self.follows[i].position, self.follows[i].direction, 
                                            np.array([self.follows[i].distance_to_leader], dtype=np.float32))))
        
        obs = np.concatenate((leader_state, *follows_state))

        return obs


    def step(self, action):
        # 리더 드론 이동   
        # 리더 드론의 위치를 무작위로 설정 (-1.0에서 1.0 사이의 무작위 값)
        if self.current_step % 100 == 0:
            self.leader_velocity = [random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)]
            
        # 리더 드론을 속도에 따라 위치 업데이트
        new_position = np.array(self.leader.position) + np.array(self.leader_velocity)
        self.leader.move(new_position)
        
        # 팔로워 드론들은 행동(action)에 따라 이동
        self.follows[0].move(action)
        
        # 팔로워 드론들이 리더와의 거리를 계산
        for i in range(NUM_FOLLOWS):
            self.follows[i].calculate_distance_to_leader(self.leader.position)
        
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
        reward += sum(self._calculate_distance_offset())
        
        # 세이프 거리 내에서 패널티
        reward += sum(self._calculate_safe_distance())        
        # 충돌 감지 패널티
        reward += sum(self._check_collision())
        
        # 너무 멀리 있는지 확인 패널티
        reward += sum(self._check_too_far())

        return reward
    
    def _calculate_distance_offset(self):

        # 팔로워 드론 1과 2의 목표 오프셋 위치 계산       
        target_positions = []
        for i in range(NUM_FOLLOWS):
            target_positions.append(self.follows[i].calculate_offset(self.leader.direction) + self.leader.position)

        # 실제 팔로워 드론 1과 2의 위치와 목표 위치 간의 거리 계산
        dist_follows = []
        for i in range(NUM_FOLLOWS):
            dist_follows.append(np.linalg.norm(self.follows[i].position - target_positions[i]))

        rewards = []
        for i in range(NUM_FOLLOWS):
            rewards.append(max(0, 1 - dist_follows[i] / MAX_DISTANCE))
            
        return rewards
    
    def _calculate_safe_distance(self):
           
        rewards = []
        
        for i in range(NUM_FOLLOWS):
            if self.follows[i].distance_to_leader < SAFE_DISTANCE:
                rewards.append(-1)
            else:
                rewards.append(0)

        return rewards
    
    def _check_collision(self):
        rewards = []
        
        # 팔로워 드론들 간 충돌
        # distance_between_followers = np.linalg.norm(self.follower_drone_1.position - self.follower_drone_2.position)
        
        # if distance_between_followers < COLLISION_DISTANCE:
        #     reward -= 10  # 충돌 시 패널티 부여
            
        # 리더 드론과 충돌
        for i in range(NUM_FOLLOWS):
            if self.follows[i].distance_to_leader < COLLISION_DISTANCE:
                rewards.append(-10)
            else:
                rewards.append(0)

        return rewards

    def _check_too_far(self):

        # 팔로워 드론 1과 2가 리더 드론으로부터 너무 멀어졌는지 확인
        rewards = []
        for i in range(NUM_FOLLOWS):
            if self.follows[i].distance_to_leader > 100:
                rewards.append(-100)
            else:
                rewards.append(0)

        return rewards
    
    def _is_done(self):
        # 리더와 팔로워가 너무 멀어지면 종료
        dist_follows = []
        for i in range(NUM_FOLLOWS):
            dist_follows.append(np.linalg.norm(self.leader.position - self.follows[i].position))
        
        if any(dist > 100 for dist in dist_follows) or self.current_step > MAX_STEPS:
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

loaded_model = PPO.load("one_agent/test.zip")
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