from abc import ABC, abstractmethod
from data_models.data_models import AgentState, LLMAnswer, NextActionDecision

class BaseAgentAction(ABC):   

    @abstractmethod
    def invoke(next_action:NextActionDecision, agent_state:AgentState) -> LLMAnswer:
        pass 

class AgentAction(BaseAgentAction):

    def __init__(self, actions:dict):
        super().__init__()
        self.actions = actions

    def invoke(self, next_action:NextActionDecision, agent_state: AgentState) -> LLMAnswer:
        
        if next_action == None:
            return None

        result = self.actions[next_action.action].invoke(agent_state)
        return result