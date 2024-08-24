import logging

import gymnasium as gym
from stable_baselines3.common.env_checker import check_env

import gym_wumpus.envs
from envs.wumpus.config import DEFAULT_WUMPUS_WORLD_CONFIG, ConfigField

logger = logging.getLogger(__name__)


def test_env_build():
    logger.info("Test VENV")
    env = gym_wumpus.envs.WumpusEnv()


def test_gym_env():
    logger.info("test_env_sample")

    env_cfg = DEFAULT_WUMPUS_WORLD_CONFIG
    env_cfg[ConfigField.PIT_PROB.name] = 0.2

    # Create environment
    env = gym.make('gym_wumpus/WumpusEnv-v0', render_mode="ansi", config=env_cfg)

    # Check the environment interface
    check_env(env, warn=True)

    # create a new instance of taxi, and get the initial state
    observation, info = env.reset()
    logging.info("Observation space {0}".format(env.observation_space))

    # Run a sample test
    terminated = False
    truncated = False
    while not terminated and not truncated:
        # print(f"step: {s} out of {num_steps}")

        # sample a random action from the list of available actions
        action = env.action_space.sample()

        # perform this action on the environment
        observation, step_reward, terminated, truncated, info = env.step(action)
        logger.info("Step reward {0}".format(step_reward))

        # print the new state
        env.render()

    # end this instance of the taxi environment
    env.close()

    # Test
    # env.unwrapped.max_possible_reward(num_steps)
    # env.unwrapped.render_all(30)


if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    test_env_build()
    test_gym_env()
