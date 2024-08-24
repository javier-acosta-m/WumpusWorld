from enum import Enum


class Orientation(Enum):
    E = 0
    S = 1
    W = 2
    N = 3

    def symbol(self) -> str:
        match self:
            case Orientation.E:
                return '>'
            case Orientation.S:
                return 'v'
            case Orientation.W:
                return '<'
            case Orientation.N:
                return '^'

    def turn_right(self) -> 'Orientation':
        match self:
            case Orientation.E:
                return Orientation.S
            case Orientation.S:
                return Orientation.W
            case Orientation.W:
                return Orientation.N
            case Orientation.N:
                return Orientation.E

    def turn_left(self) -> 'Orientation':
        match self:
            case Orientation.E:
                return Orientation.N
            case Orientation.N:
                return Orientation.W
            case Orientation.W:
                return Orientation.S
            case Orientation.S:
                return Orientation.E
        