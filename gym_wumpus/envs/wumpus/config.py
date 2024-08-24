from enum import Enum


class ConfigField(Enum):
    PIT_PROB = 'pit_prob'
    ALLOW_CLIMB_WITHOUT_GOLD = 'allow_climb_without_gold'
    HAS_WUMPUS = 'has_wumpus'
    MAX_STEPS = 'max_steps'
    GRID_SIZE = 'grid_size'


DEFAULT_WUMPUS_WORLD_CONFIG = {
    ConfigField.PIT_PROB.name: 0.0,
    ConfigField.ALLOW_CLIMB_WITHOUT_GOLD.name: False,
    ConfigField.HAS_WUMPUS.name: True,
    ConfigField.MAX_STEPS.name: 999,
    ConfigField.GRID_SIZE.name: 4
}
