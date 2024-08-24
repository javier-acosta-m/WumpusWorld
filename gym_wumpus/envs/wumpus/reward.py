from dataclasses import dataclass


@dataclass
class Reward:
    take_action = -1
    killed_by_wumpus = -1000
    fall_into_pit = -1000
    use_arrow = -10
    climb_with_gold = 1000
    grab_gold = 500
