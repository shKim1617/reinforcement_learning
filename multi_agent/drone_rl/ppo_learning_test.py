import numpy as np
import torch
import random
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import os

REL_POS_FOLLOWER1 = [-20, 20]
REL_POS_FOLLOWER2 = [-20, -20]

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
        self.terminated = False
        
        self.fig, self.ax = plt.subplots(221)  # 플롯을 위한 figure와 axis 생성
        self.drone_points = None

    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.leader.position = np.array([0.0, 0.0], dtype=np.float32)
        self.follower1.position = np.array(REL_POS_FOLLOWER1, dtype=np.float32)
        self.follower2.position = np.array(REL_POS_FOLLOWER2, dtype=np.float32)
        self.current_step = 0
        random_angle = random.uniform(-np.pi, np.pi)
        self.leader.orientation = np.array([np.cos(random_angle), np.sin(random_angle)], dtype=np.float32)
        self.leader_velocity = random.uniform(0.5, 2.0)
        
        # 창을 다시 열지 않고, 초기화만 진행
        self.ax.clear()  # 기존 창을 유지하면서 그리기 영역만 초기화
        self.ax.set_xlim(-500, 500)
        self.ax.set_ylim(-500, 500)
        
        # Return both the observation and an empty info dict
        return self._get_observation(), {}
        
    def _get_observation(self):
        distance1, direction1 = self.follower1.get_relative_distance_and_direction(self.leader)
        distance2, direction2 = self.follower2.get_relative_distance_and_direction(self.leader)
        return np.concatenate([[distance1], direction1, [distance2], direction2]).astype(np.float32)

    # 일단 스텝에서 뭘 하는지부터 좀 보자고
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
        
        self.current_step += 1
        
        self.terminated = self._check_done()
        truncated = self.current_step >= self.max_steps
        
        # 현재 상태를 화면에 그리는 함수 호출
        self.render()
        
        return self._get_observation(), reward, self.terminated, truncated, {}

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
            
            reward = - (distance_follower1 + distance_follower2)        # 거리 잘못재면 뭐라해주고
            
            # 상을 줍시다
            # 더 많은 단계를 진행하면 상을 줌
            reward += self.current_step // 100 * 10
            
            return reward

    def _check_collision(self):
        distance_between_leader_and_follower1 = np.linalg.norm(self.leader.position - self.follower1.position)
        distance_between_leader_and_follower2 = np.linalg.norm(self.leader.position - self.follower2.position)
        distance_between_followers = np.linalg.norm(self.follower1.position - self.follower2.position)
        if distance_between_followers < 2 or distance_between_leader_and_follower1 < 2 or distance_between_leader_and_follower2 < 2:
            return True
        else:
            return False

    # 이게 보상 조건에만 사용되고 종료 조건에 없었음
    def _check_drift(self):
        distance_between_leader_and_follower1 = np.linalg.norm(self.leader.position - self.follower1.position)
        distance_between_leader_and_follower2 = np.linalg.norm(self.leader.position - self.follower2.position)

        if distance_between_leader_and_follower2 > 10 or distance_between_leader_and_follower1 > 10:
            return True
        else:
            return False

    # 리더를 잃어버리면 에피소드 종료.
    # 확실히 불필요한 에피소드 경우가 줄었음
    def check_loss_leader(self):
        distance1, direction1 = self.follower1.get_relative_distance_and_direction(self.leader)
        distance2, direction2 = self.follower2.get_relative_distance_and_direction(self.leader)
        
        if distance1 > 50 or distance2 > 50:
            return True
        else:
            return False

    # 여기에 종료 조건을 좀 합쳐봐
    def _check_done(self):
        if self._check_collision() or self.check_loss_leader():
            return True
        else:
            return False
        
    def render(self, mode='human'):
        if self.drone_points is None:
            self.drone_points, = self.ax.plot([], [], 'bo')  # 드론 위치를 그릴 점
            self.ax.set_xlim(-50, 50)
            self.ax.set_ylim(-50, 50)

        self.drone_points.set_data([self.leader.position[0], self.follower1.position[0], self.follower2.position[0]],
                                   [self.leader.position[1], self.follower1.position[1], self.follower2.position[1]])
        plt.pause(0.001)  # 플롯을 업데이트하고 잠시 멈춤
   
   
# matplotlib 설정
is_ipython = 'inline' in plt.get_backend()
if is_ipython:
    from IPython import display

plt.ion()


def plot_durations(episode_durations, show_result=False):
    plt.subplot(222)
    plt.plot()
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
        
class SaveLogsCallback(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super(SaveLogsCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # 'dones' 배열에서 종료 상태를 확인
        if self.locals['dones'][0]:
            # 에피소드가 종료되면 에피소드 보상 및 길이를 기록
            episode_info = self.locals['infos'][0].get('episode', {})
            if episode_info:
                self.episode_rewards.append(episode_info.get('r', 0))
                self.episode_lengths.append(episode_info.get('l', 0))
                
                # 로그 파일로 저장
                with open(self.log_dir + "/training_log.txt", "a") as log_file:
                    log_file.write(f"Episode Reward: {episode_info.get('r', 0)}, Episode Length: {episode_info.get('l', 0)}\n")

        return True

    def _on_training_end(self) -> None:
        # 학습이 끝난 후 결과를 로그 파일로 저장
        with open(self.log_dir + "/final_results.txt", "w") as log_file:
            log_file.write(f"Final Episode Rewards: {self.episode_rewards[-1]}\n")
            log_file.write(f"Final Episode Lengths: {self.episode_lengths[-1]}\n")

## main
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = DroneFormationEnv()
seed = np.random.randint(0, 1000000)
env.reset(seed=seed)

model = PPO(
        "MlpPolicy", 
        env,
        verbose=1,
        device=device
    )

## callback
plotting_callback = PlottingCallback()

log_dir = "./drone_rl/logs"  # 로그 파일이 저장될 폴더 경로
with open("./drone_rl/logs/training_log.txt", "w") as log_file:
    pass  
save_logs_callback = SaveLogsCallback(log_dir)



callback_list = CallbackList([plotting_callback, save_logs_callback])

model.learn(total_timesteps=500, callback=callback_list)

model.save("drone_rl/model00")

# 여기에 무슨 애니메이션을 만들거냐면 학습이 어떻게 이루어지고 있는지 볼거여
print('Complete')
plot_durations(plotting_callback.episode_durations, show_result=True)
plt.ioff()
plt.show()
