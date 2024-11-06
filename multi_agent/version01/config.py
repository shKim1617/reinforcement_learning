# 팔로워 드론 수
NUM_FOLLOWS = 2
REL_POS_FOLLOWERS = [[-3.54, 3.54], [-3.54, -3.54]]
# [-3.54, 3.54] 대각선 길이 = 5

# 리더 주위
SAFE_DISTANCE = 2.5

# 오프셋 기준
MAX_DISTANCE_TO_OFFSET = 3

# 충돌 확정 범위
COLLISION_DISTANCE = 1

MAX_STEPS = 1000

TOTAL_TIMESTEPS = 1000000

# 학습 모델 경로
SAVE_MODEL_PATH = "multi_agent/version01/test"
LOAD_MODEL_PATH = "multi_agent/version01/test.zip"
SAVE_BEST_MODEL_PATH = "multi_agent/version01/best_model/"
LOAD_BEST_MODEL_PATH = "multi_agent/version01/best_model/best_model.zip"
LOG_PATH = "multi_agent/version01/logs/"