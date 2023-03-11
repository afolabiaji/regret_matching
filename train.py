from constants import NUM_ITERATIONS, NUM_ACTIONS, action_dict
from agent import Agent
import numpy as np
from tqdm import tqdm
import pickle
from pathlib import Path

def train(episodes: int):
    agent_1 = Agent()
    agent_2 = Agent()

    log = {
        "agent_1": {
            "strategy": np.zeros(shape=(episodes, NUM_ACTIONS)),
            "regret": np.zeros(shape=(episodes, NUM_ACTIONS)),
        },
        "agent_2": {
            "strategy": np.zeros(shape=(episodes, NUM_ACTIONS)),
            "regret": np.zeros(shape=(episodes, NUM_ACTIONS)),
        },
    }

    for i in tqdm(range(episodes)):
        if ((i + 1) % 10_000) == 0:
            print(f"Training {i+1}th iteration")

        action_1 = agent_1.get_action()
        action_2 = agent_2.get_action()

        agent_1.update_regret(action_1, action_2)
        agent_2.update_regret(action_2, action_1)

        agent_1.update_strategy_from_regret()
        agent_2.update_strategy_from_regret()
        
        log["agent_1"]["strategy"][i] = agent_1.strategy
        log["agent_2"]["strategy"][i] = agent_2.strategy

        log["agent_1"]["regret"][i] = agent_1.cumulative_regret
        log["agent_2"]["regret"][i] = agent_2.cumulative_regret

        if ((i + 1) % 10_000) == 0:
            print(
                f"Agent 1 Action: {action_dict[agent_1.action[0]]}, \n" 
                f"Agent 2 Action: {action_dict[agent_2.action[0]]}, \n"
                f"Agent 1 Regret: {agent_1.cumulative_regret}, \n"
                f"Agent 1 Utility: {agent_1.utility}, \n"
                f"Agent 1 Cumulative Utility: {agent_1.cumulative_utility}, \n"
                f"Agent 1 Strategy: {agent_1.strategy} \n"
                f"Agent 1 Cumulative Strategy: {agent_1.get_average_strategy()}"
            )
    
    mod_path = Path(__file__).parent
    src_path = (mod_path / "./logs/agent_log.pkl").resolve()
    with open(src_path, "wb") as file:
        pickle.dump(log, file)

    print(f"Average Strategy: {agent_1.get_average_strategy()}")