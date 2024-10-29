import numpy as np
import torch
import random
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, BaseCallback
from drone_class import create_drones
from values import NUM_FOLLOWS, MAX_DISTANCE, MAX_STEPS, SAFE_DISTANCE, COLLISION_DISTANCE

# 환경 설정
class DroneFormationEnv(gym.Env):
    def __init__(self):
        super(DroneFormationEnv, self).__init__()
        
        # 상태 공간: 리더 위치, 방향 + 팔로워 위치, 방향, 리더와의 거리 = 4 + 5
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        
        # 행동 공간: 팔로워 드론들의 이동 방향 (예: [dx, dy]로 각 드론의 이동을 정의)
        self.action_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(2,), dtype=np.float32)
        
        
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
        target_leader_position = np.array(self.leader.position) + np.array(self.leader_velocity)
        self.leader.move(target_leader_position)
        
        # 팔로워 드론들은 행동(action)에 따라 이동
        follw_target_position = self.follows[0].position + action
        self.follows[0].move(follw_target_position)
        
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