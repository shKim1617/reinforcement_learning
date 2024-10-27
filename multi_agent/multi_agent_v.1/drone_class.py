import numpy as np

class Drone:
    def __init__(self, drone_id, init_position=[0, 0], init_direction=[1, 0]):
        self.drone_id = drone_id
        self.position = np.array(init_position, dtype=np.float32)  # 드론의 위치 (x, y)
        self.direction = np.array(init_direction, dtype=np.float32)  # 드론의 초기 방향
        
    def move(self, target_position):
        """
        드론이 목표 위치로 이동하고 그 방향으로 회전하는 메소드
        :param target_position: 이동할 목표 좌표 (x, y)
        """
        self.update_direction(target_position)
        self.set_position(target_position)
        
        
    def set_position(self, position):
        """
        드론의 위치를 설정하는 메소드
        """
        self.position = np.array(position, dtype=np.float32)
    
    def update_direction(self, target_position):
        """
        드론이 이동한 방향을 기준으로 방향을 설정하는 메소드
        :param target_position: 목표 위치 (x, y)
        """
        # 목표 위치로의 방향 벡터 계산
        direction_vector = np.array(target_position) - self.position
        # print(direction_vector)
        
        # 방향 벡터의 크기를 계산
        norm = np.linalg.norm(direction_vector)
        
        # 벡터의 크기가 0이 아니면 정규화 (방향 설정), 0이면 방향 유지
        if norm != 0:
            self.direction = direction_vector / norm
    
    def get_state(self):
        """
        드론의 현재 상태를 반환 (위치와 방향)
        """
        return self.position, self.direction


# 예시: 리더 드론 생성
drone = Drone(drone_id=0, init_position=[0, 0])

print(drone.get_state())  # 초기 상태 확인

# 드론의 목표 위치 설정
new_position = [0, 0]  # 이동할 위치

# 드론 이동 및 방향 업데이트
drone.move(new_position)

# 드론 상태 확인
print(drone.get_state())  # 최종 상태 확인
