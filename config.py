# Training length and reproducibility.
EPISODES = 2_000
SEED = 42

# Replay buffer and sampling settings.
REPLAY_CAPACITY = 100_000
WARMUP_STEPS = 20_000
BATCH_SIZE = 32

# DQN learning settings.
GAMMA = 0.99
LEARNING_RATE = 0.00005
EPSILON_START = 1.0
EPSILON_FINAL = 0.05
EPSILON_DECAY_STEPS = 150_000
TARGET_UPDATE_STEPS = 10_000
MAX_ABS_Q_VALUE = 100.0

# Evaluation schedule.
EVALUATION_INTERVAL = 100
EVALUATION_EPISODES = 20

# Output files.
MODEL_PATH = "dqn_assault.weights_1.h5"
BEST_MODEL_PATH = "dqn_assault_best.weights_1.h5"
HISTORY_PATH = "training_history_assault_1.csv"
