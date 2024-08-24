import logging
import random
import threading
from typing import SupportsFloat, Any, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType

import gym_wumpus
from .wumpus.action import Action
from .wumpus.config import DEFAULT_WUMPUS_WORLD_CONFIG, ConfigField
from .wumpus.location import Location
from .wumpus.orientation import Orientation
from .wumpus.reward import Reward
from .wumpus.stats_manager import StatsEntryRow, StatsManager

INFO_TOTAL_REWARD = 'total_reward'

logger = logging.getLogger(gym_wumpus.LOGGER_NAME)


class Counter:
    __id: int = 0
    __lock = threading.Lock()

    @staticmethod
    def next_id():
        with Counter.__lock:
            next_id_value = Counter.__id
            Counter.__id += 1
        return next_id_value


class WumpusEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, config=DEFAULT_WUMPUS_WORLD_CONFIG, render_mode=None, stat_manager: StatsManager = None):
        super().__init__()
        # Pre-conditions
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self._id = Counter.next_id()
        self._config = config
        self.stat_manager = stat_manager
        self.render_mode = render_mode

        # Extract config
        self.pit_prob = self._config[ConfigField.PIT_PROB.name]
        self.allow_climb_without_gold = self._config[ConfigField.ALLOW_CLIMB_WITHOUT_GOLD.name]
        self.has_wumpus = self._config[ConfigField.HAS_WUMPUS.name]
        self._max_steps = self._config[ConfigField.MAX_STEPS.name]
        self.grid_size = self._config[ConfigField.GRID_SIZE.name]

        # Setup environment initial conditions
        self.wumpus_location = None
        self.wumpus_alive = None
        self.gold_location = None
        self.pit_locations = None
        self.agent_has_gold = None
        self.agent_has_arrow = None
        self.time_step = 0

        # Build environment
        self.agent_location = Location(0, 0, self.grid_size)
        self.agent_orientation = Orientation.E
        self.__place_gold()
        self.__place_pits()
        if self.has_wumpus:
            self.__place_wumpus()

        # Set agent state
        self.agent_state_location = np.zeros(self.grid_size * self.grid_size)
        self.agent_state_stench_locations = np.zeros(self.grid_size * self.grid_size)
        self.agent_state_breeze_locations = np.zeros(self.grid_size * self.grid_size)
        self.agent_state_visited_locations = np.zeros(self.grid_size * self.grid_size)
        self.agent_state_orientation = np.array([1., 0., 0., 0.])
        self.agent_state_heard_scream = np.array([0.0])
        self.agent_state_has_gold = np.array([0.0])
        self.agent_state_has_arrow = np.array([1.0])
        self.agent_state_sense_glitter = np.array([0.0])
        self.agent_state_sense_bump = np.array([0.0])

        # To track statistics
        self.stats_entry = StatsEntryRow()
        self.stats_entry.id = self._id
        self.info_total_reward = 0.0
        self.info_last_action = None
        self._history = []

        # Action Space
        self.action_space = gym.spaces.Discrete(len(Action))

        # Observation Space
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=self.__agent_state().shape, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        random.seed(seed)
        self.action_space.seed(int((self.np_random.uniform(0, seed if seed is not None else 1))))

        # Setup environment initial conditions
        self.wumpus_location = None
        self.gold_location = None
        self.pit_locations = None
        self.wumpus_alive = False
        self.agent_has_gold = False
        self.agent_has_arrow = True
        self.time_step = 0

        # Build environment
        self.agent_location = Location(0, 0, self.grid_size)
        self.agent_orientation = Orientation.E
        self.__place_gold()
        self.__place_pits()
        if self.has_wumpus:
            self.__place_wumpus()
            self.wumpus_alive = True

        # Set agent state
        self.agent_state_location = np.zeros(self.grid_size * self.grid_size)
        self.agent_state_stench_locations = np.zeros(self.grid_size * self.grid_size)
        self.agent_state_breeze_locations = np.zeros(self.grid_size * self.grid_size)
        self.agent_state_visited_locations = np.zeros(self.grid_size * self.grid_size)
        self.agent_state_orientation = np.array([1., 0., 0., 0.])
        self.agent_state_heard_scream = np.array([0.0])
        self.agent_state_has_gold = np.array([0.0])
        self.agent_state_has_arrow = np.array([1.0])
        self.agent_state_sense_glitter = np.array([0.0])
        self.agent_state_sense_bump = np.array([0.0])

        # To track statistics
        self.stats_entry = StatsEntryRow()
        self.stats_entry.id = self._id
        self.info_total_reward = 0.0
        self.info_last_action = None
        self._history = []

        if self.render_mode == 'human':
            pass
        elif self.render_mode == "ansi":
            pass

        observation = self.__get_obs()
        info = self.__get_info()

        return observation, info

    def step(self, action) -> Tuple[ObsType, float, bool, dict]:
        # Local variables
        step_reward = 0
        heard_scream = False
        sense_bump = False

        # Episode variables
        truncated = False  # Whether the truncation condition outside the scope of the MDP is satisfied
        terminated = False  # Whether the agent reaches the terminal state

        # Check end of episode
        if self.time_step > self._max_steps:
            truncated = True
        else:
            step_reward = Reward.take_action
            self.stats_entry.action_count[action] += 1
            action_enum = Action.from_int(action)
            match action_enum:
                case Action.LEFT:
                    self.agent_orientation = self.agent_orientation.turn_left()

                case Action.RIGHT:
                    self.agent_orientation = self.agent_orientation.turn_right()

                case Action.FORWARD:
                    sense_bump = self.agent_location.forward(self.agent_orientation)
                    if self.__is_pit_at(self.agent_location):
                        self.stats_entry.killed_by_pit = 1
                        logger.debug("{0}: killed BY PIT".format(self._id))
                        step_reward += Reward.fall_into_pit
                        terminated = True
                    if self.wumpus_alive and self.__is_wumpus_at(self.agent_location):
                        logger.info("{0}: killed BY WUMPUS".format(self._id))
                        self.stats_entry.killed_by_wumpus = 1
                        step_reward += Reward.killed_by_wumpus
                        terminated = True

                case Action.GRAB:
                    if self.agent_has_gold:
                        pass
                    elif self.agent_location.is_location(self.gold_location):
                        self.stats_entry.steps_retrieve_gold = self.time_step
                        self.agent_has_gold = True
                        step_reward += Reward.grab_gold
                        logger.debug("{0}: grabbed GOLD".format(self._id))

                case Action.SHOOT:
                    if self.agent_has_arrow:
                        logger.debug("{0}: used ARROW".format(self._id))
                        heard_scream = self.__kill_wumpus_attempt()
                        step_reward += Reward.use_arrow
                        self.agent_has_arrow = False

                case Action.CLIMB:
                    # Valid only at (0,0)
                    if self.agent_location.is_location(Location(0, 0, self.grid_size)):

                        # Check if the agent has the gold
                        if self.agent_has_gold:
                            step_reward = Reward.climb_with_gold
                            self.stats_entry.exit_with_gold = 1
                            self.stats_entry.steps_exit = self.time_step - self.stats_entry.steps_retrieve_gold
                            logger.info("*** {}: WIN ***".format(self._id))

                        elif self.allow_climb_without_gold:
                            self.stats_entry.exit_without_gold = 1
                            logger.info("\t{0}: Climbed WITHOUT GOLD".format_map(self._id))

                        # Check end-episode condition
                        if self.allow_climb_without_gold or self.agent_has_gold:
                            terminated = True

        # Update display
        if self.agent_has_gold:
            self.gold_location.set_to(self.agent_location)

        # Update info
        self.info_total_reward += step_reward
        self.info_last_action = action
        self._history.append({'action': action, 'step_reward': step_reward})

        # Update agent believe state after the action
        self.__update_agent_believe_state(heard_scream, sense_bump)

        # Get  the new env observations
        observation = self.__get_obs()
        info = self.__get_info()

        # Increase time step
        self.time_step = self.time_step + 1

        # Track stats
        done = (terminated or truncated)
        if done and self.stat_manager:
            self.stats_entry.reward = self.info_total_reward
            self.stat_manager.add_stats(self.stats_entry)

        return observation, step_reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            pass
        elif self.render_mode == "ansi":
            ansi_retval = ""
            ansi_retval += "step:{0},".format(self.time_step - 1)
            ansi_retval += "action:{0},".format(Action.from_int(self.info_last_action).name)
            ansi_retval += "has arrow:{0},".format(self.agent_has_arrow)
            ansi_retval += "heard_scream:{0}".format(self.agent_state_heard_scream[0])
            ansi_retval += "sum(reward):{0}".format(self.info_total_reward)
            ansi_retval += "\n"
            for row in range(self.grid_size - 1, -1, -1):
                line = '{}: |'.format(row)
                for col in range(0, self.grid_size):
                    loc = Location(row, col, grid_size=self.grid_size)
                    loc_idx = loc.to_linear()
                    cell_symbols = [' '] * 7
                    # Draw agent (0)
                    if self.__is_agent_at(loc):
                        cell_symbols[0] = self.agent_orientation.symbol()
                    # Draw pit (1)
                    if self.__is_pit_at(loc):
                        cell_symbols[1] = 'P'
                    # Draw WUMPUS (2)
                    if self.has_wumpus and self.__is_wumpus_at(loc):
                        if self.wumpus_alive: cell_symbols[2] = 'W'
                        else: cell_symbols[2] = 'w'

                    # Draw gold (3)
                    if self.__is_gold_at(loc) or self.agent_has_gold:
                        cell_symbols[3] = 'G'

                    # Draw breeze
                    if self.agent_state_breeze_locations[loc_idx] == 1.0:
                        cell_symbols[4] = '~'

                    # Draw stench
                    if self.agent_state_stench_locations[loc_idx] == 1.0:
                        cell_symbols[5] = 's'

                    # Draw visited
                    if self.agent_state_visited_locations[loc_idx] == 1.0:
                        cell_symbols[6] = '*'

                    for char in cell_symbols:
                        line += char
                    line += '|'
                ansi_retval += line + "\n"
            logger.info(ansi_retval)
            return ansi_retval

    def close(self):
        pass

    def __agent_state(self):
        return np.concatenate((
            self.agent_state_location,
            self.agent_state_orientation,
            self.agent_state_visited_locations,
            self.agent_state_stench_locations,
            self.agent_state_breeze_locations,
            self.agent_state_heard_scream,
            self.agent_state_has_gold,
            self.agent_state_has_arrow,
            self.agent_state_sense_glitter,
            self.agent_state_sense_bump,
        )).astype(np.float32)

    def __get_obs(self):
        obs = self.__agent_state()
        return obs

    def __get_info(self):
        return {
            INFO_TOTAL_REWARD: self.info_total_reward,
        }

    def __update_agent_believe_state(self, heard_scream, sense_bump):
        # Clear the location & set
        self.agent_state_location = np.zeros(self.grid_size * self.grid_size)
        self.agent_state_location[self.agent_location.to_linear()] = 1.0

        # Set the agent orientation
        match self.agent_orientation:
            case Orientation.E:
                self.agent_state_orientation = np.array([1., 0., 0., 0.])
            case Orientation.S:
                self.agent_state_orientation = np.array([0., 1., 0., 0.])
            case Orientation.W:
                self.agent_state_orientation = np.array([0., 0., 1., 0.])
            case Orientation.N:
                self.agent_state_orientation = np.array([0., 0., 0., 1.])

        # Set the visited locations
        self.agent_state_visited_locations[self.agent_location.to_linear()] = 1.0

        # Sense WUMPUS stench
        if self.__sense_stench():
            self.agent_state_stench_locations[self.agent_location.to_linear()] = 1.0

        # Sense PIT's breeze
        if self.__sense_breeze():
            self.agent_state_breeze_locations[self.agent_location.to_linear()] = 1.0

        # Sense gold glitter
        if self.__sense_glitter():
            self.agent_state_sense_glitter[0] = 1.0
        else:
            self.agent_state_sense_glitter[0] = 0.0

        # Heard scream from WUMPUS
        if heard_scream:
            self.agent_state_heard_scream[0] = 1.0

        # Sense BUMP from the wall
        if sense_bump:
            self.agent_state_sense_bump[0] = 1.0
        else:
            self.agent_state_sense_bump[0] = 0.0

        # Has the arrow
        if self.agent_has_arrow:
            self.agent_state_has_arrow[0] = 1.0
        else:
            self.agent_state_has_arrow[0] = 0.0

        # Has the gold
        if self.agent_has_gold:
            self.agent_state_has_gold[0] = 1.0
        else:
            self.agent_state_has_gold[0] = 0.0

    def __place_wumpus(self):
        self.wumpus_alive = True
        self.wumpus_location = Location.random(self.grid_size)

    def __place_gold(self):
        self.gold_location = Location.random(self.grid_size)

    def __place_pits(self):
        self.pit_locations = []
        for i in range(1, self.grid_size * self.grid_size):
            if random.random() < self.pit_prob:
                self.pit_locations.append(Location.from_linear(i, self.grid_size))

    def __is_pit_at(self, location: Location) -> bool:
        return any(pit_location.is_location(location) for pit_location in self.pit_locations)

    def __is_pit_adjacent_to_agent(self) -> bool:
        for agent_neighbour in self.agent_location.neighbours():
            for pit_location in self.pit_locations:
                if agent_neighbour.is_location(pit_location):
                    return True
        return False

    def __is_wumpus_adjacent_to_agent(self) -> bool:
        return self.has_wumpus and any(self.wumpus_location.is_location(neighbour) for neighbour in self.agent_location.neighbours())

    def __is_agent_at_hazard(self) -> bool:
        return self.__is_pit_at(self.agent_location) or (self.is_wumpus_at(self.agent_location) and self.wumpus_alive)

    def __is_wumpus_at(self, location: Location) -> bool:
        retval = False
        if self.has_wumpus:
            retval = self.wumpus_location.is_location(location)
        return retval

    def __is_agent_at(self, location: Location) -> bool:
        return self.agent_location.is_location(location)

    def __is_gold_at(self, location: Location) -> bool:
        return self.gold_location.is_location(location)

    def __sense_glitter(self) -> bool:
        if self.agent_has_gold:
            return False
        else:
            return self.__is_gold_at(self.agent_location)

    def __sense_breeze(self) -> bool:
        return self.__is_pit_adjacent_to_agent()

    def __sense_stench(self) -> bool:
        return self.__is_wumpus_adjacent_to_agent()

    def __wumpus_in_line_of_fire(self) -> bool:
        match self.agent_orientation:
            case Orientation.E:
                return self.has_wumpus and self.agent_location.is_left_of(self.wumpus_location)
            case Orientation.S:
                return self.has_wumpus and self.agent_location.is_above(self.wumpus_location)
            case Orientation.W:
                return self.has_wumpus and self.agent_location.is_right_of(self.wumpus_location)
            case Orientation.N:
                return self.has_wumpus and self.agent_location.is_below(self.wumpus_location)

    def __kill_wumpus_attempt(self) -> np.array:
        heard_scream = np.array([0.0])
        if self.has_wumpus and self.wumpus_alive:
            if self.__wumpus_in_line_of_fire():
                logger.info("{0}: killed WUMPUS".format(self._id))
                self.wumpus_alive = False
                heard_scream = np.array([1.0])
        return heard_scream


if __name__ == "__main__":
    print("Test")
