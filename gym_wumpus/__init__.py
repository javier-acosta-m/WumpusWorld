import gymnasium as gym

LOGGER_NAME = "gym_wumpus"

gym.envs.register(
    id="gym_wumpus/WumpusEnv-v0",
    entry_point="gym_wumpus.envs:WumpusEnv",
)
