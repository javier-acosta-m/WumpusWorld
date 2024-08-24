from enum import Enum
from random import choice


class Action(Enum):
    LEFT = 0
    RIGHT = 1
    FORWARD = 2
    GRAB = 3
    SHOOT = 4
    CLIMB = 5

    @staticmethod
    def random() -> 'Action':
        return choice(list(Action))

    @staticmethod
    def from_int(n: int) -> 'Action':
        return Action(n)
