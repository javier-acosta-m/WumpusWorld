import copy
import threading
from datetime import datetime
from pathlib import Path

from gym_wumpus.envs.wumpus.action import Action


class StatsEntryRow:
    def __init__(self):
        self.id = 0
        self.reward = 0
        self.action_count = [0, 0, 0, 0, 0, 0]
        self.killed_by_wumpus = 0
        self.killed_by_pit = 0
        self.exit_with_gold = 0
        self.exit_without_gold = 0
        self.steps_retrieve_gold = 0
        self.steps_exit = 0
        self.total_episodes = 0


class StatsEntryHeader:
    def __init__(self):
        self.header = ""
        self.header += "id" + ","
        self.header += "reward" + ","
        for action_type in Action:
            self.header += action_type.name + ","
        self.header += "killed_by_wumpus" + ","
        self.header += "killed_by_pit" + ","
        self.header += "exit_with_gold" + ","
        self.header += "exit_without_gold" + ","
        self.header += "steps_retrieve_gold" + ","
        self.header += "steps_exit" + ","
        self.header += "total_steps" + ","
        self.header += "total_episodes" + "\n"

    def __str__(self):
        return self.header.upper()


def save_entry(file, entry: StatsEntryRow):
    csv_row = ""
    csv_row += "{0},".format(entry.id)
    csv_row += "{0},".format(entry.reward)
    for it in entry.action_count:
        csv_row += "{0},".format(it)
    csv_row += "{0},".format(entry.killed_by_wumpus)
    csv_row += "{0},".format(entry.killed_by_pit)
    csv_row += "{0},".format(entry.exit_with_gold)
    csv_row += "{0},".format(entry.exit_without_gold)
    csv_row += "{0},".format(entry.steps_retrieve_gold)
    csv_row += "{0},".format(entry.steps_exit)
    csv_row += "{0},".format(sum(entry.action_count))
    csv_row += "{0}\n".format(entry.total_episodes)
    ascii_text = csv_row.encode('ascii')
    # print(ascii_text)
    file.write(ascii_text)


class StatsManager:
    def __init__(self, filename="./logs/stats", auto_save: bool = True, auto_save_period: int = 15):
        path = Path(filename)
        Path(path.parent).mkdir(parents=True, exist_ok=True)
        self._auto_save = auto_save
        self._auto_save_period = auto_save_period
        self._filepath = filename + '_{:%Y-%m-%d_%H:%M:%S}.csv'.format(datetime.now())
        self._entry_list_lock = threading.Lock()
        self._entry_list_current = []
        self._file_data = open(self._filepath, 'wb')

        header = StatsEntryHeader()
        ascii_text = header.__str__().encode('ascii')
        self._file_data.write(ascii_text)
        if auto_save:
            self.start_auto_save()

    def filepath(self):
        return self._filepath

    def reset(self):
        with self._entry_list_lock:
            self._entry_list_current = []

    def save(self):
        # Move the elements to the unsaved list & reset the current list
        with self._entry_list_lock:
            # print("{0} saving length is  {1}".format(hex(id(self._entry_list_current)), len(self._entry_list_current)))
            list_unsaved = copy.deepcopy(self._entry_list_current)
            self._entry_list_current.clear()
        for entry in list_unsaved:
            save_entry(self._file_data, entry)

    def add_stats(self, entry: StatsEntryRow):
        with self._entry_list_lock:
            self._entry_list_current.append(entry)
        # print("{0} {1}".format(hex(id(self._entry_list_current)), len(self._entry_list_current)))

    def start_auto_save(self):
        self._auto_save = True
        self.__save_stats_timer(self._auto_save_period)

    def stop_auto_save(self):
        self._auto_save = False

    def __save_stats_timer(self, period_sec):
        if self._auto_save:
            threading.Timer(period_sec, self.__save_stats_timer, [period_sec]).start()
            self.save()


if __name__ == '__main__':
    # Test
    stats = StatsManager("./logs/wumpus_raw")
    row = StatsEntryRow()
    row.killed_by_wumpus = 1
    row.exit_with_gold = 2
    row.exit_without_gold = 3
    row.steps_retrieve_gold = 4
    row.steps_exit = 5
    row.total_episodes = 6

    list_actions = []
    counter = 2
    for action in gym_wumpus.action.Action:
        list_actions.append(counter)
        counter += 2
    row.action_count = list_actions
    stats.add_stats(row)
    row.total_episodes = 100
    stats.add_stats(row)
    stats.save()
