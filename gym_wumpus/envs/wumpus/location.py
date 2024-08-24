from random import randint
from typing import List

from .orientation import Orientation


class Location:

    def __init__(self, row: int, col: int, grid_size):
        self.col = col
        self.row = row
        self._grid_size = grid_size

    def __str__(self):
        return '({0},{1})'.format(self.col, self.row)

    def is_left_of(self, location: 'Location') -> bool:
        return self.col < location.col and self.row == location.row

    def is_right_of(self, location: 'Location') -> bool:
        return self.col > location.col and self.row == location.row

    def is_above(self, location: 'Location') -> bool:
        return self.row > location.row and self.col == location.col

    def is_below(self, location: 'Location') -> bool:
        return self.row < location.row and self.col == location.col

    def neighbours(self) -> List['Location']:
        neighbour_list = []
        if self.col > 0:
            neighbour_list.append(Location(self.col - 1, self.row, self._grid_size))
        if self.col < self._grid_size:
            neighbour_list.append(Location(self.col + 1, self.row, self._grid_size))
        if self.row > 0:
            neighbour_list.append(Location(self.col, self.row - 1, self._grid_size))
        if self.row < self._grid_size-1:
            neighbour_list.append(Location(self.col, self.row + 1, self._grid_size))
        return neighbour_list

    def is_location(self, location: 'Location') -> bool:
        return self.col == location.col and self.row == location.row

    def at_left_edge(self) -> bool:
        return self.col == 0

    def at_right_edge(self) -> bool:
        return self.col == self._grid_size-1

    def at_top_edge(self) -> bool:
        return self.row == self._grid_size-1

    def at_bottom_edge(self) -> bool:
        return self.row == 0

    def forward(self, orientation) -> bool:
        bump = False
        match orientation:
            case Orientation.W:
                if self.at_left_edge():
                    bump = True
                else:
                    self.col = self.col - 1
            case Orientation.E:
                if self.at_right_edge():
                    bump = True
                else:
                    self.col = self.col + 1
            case Orientation.N:
                if self.at_top_edge():
                    bump = True
                else:
                    self.row = self.row + 1
            case Orientation.S:
                if self.at_bottom_edge():
                    bump = True
                else:
                    self.row = self.row - 1
        return bump

    def set_to(self, location: 'Location'):
        self.row = location.row
        self.col = location.col

    def to_linear(self) -> int:
        n = (self.row * self._grid_size) + self.col
        return n

    @staticmethod
    def from_linear(n: int, grid_size: int) -> 'Location':
        row = n % grid_size
        col = n // grid_size
        return Location(row, col, grid_size=grid_size)

    @staticmethod
    def random(grid_size) -> 'Location':
        n = randint(1, (grid_size*grid_size)-1)
        return Location.from_linear(n, grid_size)
