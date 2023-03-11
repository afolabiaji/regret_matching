from constants import NUM_ACTIONS
import numpy as np

class Agent:
    def __init__(self, num_actions=NUM_ACTIONS, strategy=None):
        if strategy is None:
            self.strategy = np.repeat(1 / num_actions, num_actions)
        else:
            self.strategy = strategy

        self.num_actions = num_actions
        self.cumulative_strategy = self.strategy
        self.cumulative_regret = np.repeat(0, num_actions)
        self.cumulative_utility = 0
        
    def get_action(self) -> int:
        self.action = np.random.choice(int(self.num_actions), 1, p=self.strategy)
        return self.action[0]

    def update_regret(self, action_1: int, action_2: int):
        action_utility = np.repeat(0, self.num_actions)

        action_utility[action_2] = 0

        if action_2 == 2:
            action_utility[0] = 1
        else:
            action_utility[(action_2 + 1)] = 1

        if action_2 == 0:
            action_utility[2] = -1
        else:
            action_utility[(action_2 - 1)] = -1
        
        self.utility = action_utility[action_1]
        self.cumulative_utility = self.cumulative_utility + self.utility
        current_regret = (action_utility - action_utility[action_1])
        self.cumulative_regret = self.cumulative_regret + current_regret

    def update_strategy_from_regret(self) -> np.array:
        self.strategy = np.where(self.cumulative_regret > 0, self.cumulative_regret, 0)

        if sum(self.strategy) > 0:
            self.strategy = np.array(self.strategy / sum(self.strategy))
        else:
            self.strategy = np.repeat(1 / self.num_actions, self.num_actions)

        self.cumulative_strategy = self.cumulative_strategy + self.strategy

        return self.cumulative_strategy

    def get_average_strategy(self) -> np.array:
        if sum(self.cumulative_strategy) > 0:
            average_strategy = self.cumulative_strategy / sum(self.cumulative_strategy)
        else:
            average_strategy = np.repeat(1 / 3, 3)
    
        return average_strategy