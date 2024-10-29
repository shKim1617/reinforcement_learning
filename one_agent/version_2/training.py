import torch
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, BaseCallback
from env_class import DroneFormationEnv

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

model.learn(total_timesteps=1000, callback=callback_list)

model.save("one_agent/test")

print('Complete')
plot_durations(plotting_callback.episode_durations, show_result=True)
plt.ioff()
plt.show()