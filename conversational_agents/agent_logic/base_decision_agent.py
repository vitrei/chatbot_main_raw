from abc import ABC, abstractmethod

from data_models.data_models import AgentState, NextActionDecision

class BaseDecisionAgent(ABC):

    @abstractmethod
    def next_action(agent_state: AgentState) -> NextActionDecision:
        pass




        