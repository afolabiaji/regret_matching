import sys
import numpy as np
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
from agent import Agent
from constants import NUM_ACTIONS

def init_agent() -> Agent:
    agent = Agent()
    return agent

def add_actions_to_agent(agent:Agent) -> Agent:
    action_1 = 0
    action_2 = 0
    agent.update_regret(action_1, action_2)
    return agent

def test_init_strategy() -> None:
    agent = init_agent()
    assert np.array_equiv(agent.strategy, np.repeat(1/NUM_ACTIONS, NUM_ACTIONS))

def test_get_action() -> None:
    agent = init_agent()
    assert agent.get_action() in range(NUM_ACTIONS)

def test_update_regret() -> None:
    agent = init_agent()
    agent = add_actions_to_agent(agent)
    assert agent.utility == 0

def test_update_strategy_from_regret() -> None:
    agent = init_agent()
    agent = add_actions_to_agent(agent)
    agent.update_strategy_from_regret()
    paper_beats_rock_strategy = np.array([0,1,0])
    assert np.array_equiv(
        agent.cumulative_strategy, 
        paper_beats_rock_strategy + np.repeat(1/NUM_ACTIONS, NUM_ACTIONS)
    )

def test_get_average_strategy() -> None:
    agent = init_agent()
    agent = add_actions_to_agent(agent)
    average_strategy = agent.get_average_strategy()
    np.testing.assert_array_equal(
        average_strategy,
        np.array([1/3,1/3,1/3])
    )
