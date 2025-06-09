from abc import ABC, abstractmethod
from data_models.data_models import AgentState

class BasePreProcessor(ABC):

    @abstractmethod
    def invoke(self, agent_state: AgentState) -> AgentState:
        pass