import numpy as np

# 드론의 상태를 정의하는 클래스
class Drone:
    def __init__(self, position):
        self.position = np.array(position)
        self.velocity = np.array([0, 0, 0])

    def update_position(self, target_position, learning_model):
        # 딥러닝 모델을 사용해 새로운 위치를 예측하고 업데이트
        self.position = learning_model.predict(self.position, target_position)

class Formation:
    def __init__(self, leader, followers):
        self.leader = leader
        self.followers = followers

    def update_formation(self, leader_target_position, learning_model):
        # 선두 드론의 목적지 업데이트
        self.leader.update_position(leader_target_position, learning_model)

        # 팔로워 드론의 위치를 선두 드론에 맞춰 업데이트
        for i, follower in enumerate(self.followers):
            # 임시 목적지 계산 (A자 대형 유지)
            offset_angle = np.pi / 4 if i == 0 else -np.pi / 4
            follower_target_position = self.leader.position + np.array([
                np.cos(offset_angle), np.sin(offset_angle), 0]) * 10  # 거리 조정

            follower.update_position(follower_target_position, learning_model)

# 예시로 딥러닝 모델을 간단하게 시뮬레이션
class DummyModel:
    def predict(self, current_position, target_position):
        # 단순히 선형 보간을 사용하여 위치 업데이트
        return current_position + 0.1 * (target_position - current_position)

# 드론 초기화
leader = Drone([0, 0, 0])
follower1 = Drone([0, 10, 0])
follower2 = Drone([0, -10, 0])

# 대형 생성
formation = Formation(leader, [follower1, follower2])

# 딥러닝 모델 (여기서는 더미 모델을 사용)
model = DummyModel()

# 반복적으로 대형 업데이트
for step in range(100):
    leader_target = [step, 0, 0]  # 선두 드론의 목적지
    formation.update_formation(leader_target, model)

    # 각 드론의 위치 출력
    print(f"Step {step} - Leader: {leader.position}, Follower1: {follower1.position}, Follower2: {follower2.position}")
