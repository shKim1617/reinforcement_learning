import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("CartPole-v1", render_mode="rgb_array")

obs, info = env.reset(seed=42)

print(obs, info, type(obs))

# img = env.render()
# print(img.shape)

# plt.imshow(img)
# plt.show()

action = 1
obs, reward, done, truncated, info = env.step(action)
print(obs)

# img = env.render()
# plt.imshow(img)
# plt.show()

def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1

totals = []
for episode in range(500):
    episode_rewards = 0
    obs, info = env.reset(seed=episode)
    for step in range(200):
        action = basic_policy(obs)
        obs, reward, done, truncated, info = env.step(action)
        episode_rewards += reward
        if done or truncated:
            break
        
    totals.append(episode_rewards)
    
print(np.mean(totals), np.std(totals), min(totals), max(totals))