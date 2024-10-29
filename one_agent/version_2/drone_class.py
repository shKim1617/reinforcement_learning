import numpy as np
from values import SAFE_DISTANCE, REL_POS_FOLLOWERS

# 수치 정의
# NUM_FOLLOWS = 1
# REL_POS_FOLLOWERS = [[-20, 20], [20, 20]]
# SAFE_DISTANCE = 5
# MAX_DISTANCE = 30
# COLLISION_DISTANCE = 1
# MAX_STEPS = 1000

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
            

# 동적으로 팔로워 드론 생성 및 상태 출력 예시
def create_drones(num_follows):
    # 리더 드론 생성
    leader_drone = LeaderDrone()
    
    # 팔로워 드론을 num_followers 수만큼 리스트에 생성
    follower_drones = [
        FollowerDrone(drone_id=i + 1, offset=REL_POS_FOLLOWERS[i]) 
        for i in range(num_follows)
    ]
    
    # # 리더 드론 상태 출력
    # print(f"Leader Drone (ID: {leader_drone.drone_id}): Position = {leader_drone.position}, Direction = {leader_drone.direction}")
    
    # # 팔로워 드론 상태 출력
    # for follower in follower_drones:
    #     print(f"Follower Drone (ID: {follower.drone_id}): Position = {follower.position}, Offset = {follower.offset}, Distance to Leader = {follower.distance_to_leader}")
    
    return leader_drone, follower_drones

# test
# leader, followers = create_drones(NUM_FOLLOWS)
# followers[0].move([10,10])
# print(followers[0].get_state())

# print(followers[0].drone_id)