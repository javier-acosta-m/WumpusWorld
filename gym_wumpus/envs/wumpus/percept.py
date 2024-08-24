class Percept:

    def __init__(self, time_step: int, bump: bool, breeze: bool, stench: bool, scream: bool, glitter: bool, reward: int, done: bool):
        self.time_step = time_step
        self.bump = bump
        self.breeze = breeze
        self.stench = stench
        self.scream = scream
        self.glitter = glitter
        self.reward = reward
        self.done = done
        
    def __str__(self):
        return f'time:{self.time_step}: bump:{self.bump}, breeze:{self.breeze}, stench:{self.stench}, scream:{self.scream}, glitter:{self.glitter}, reward:{self.reward}, done:{self.done}'