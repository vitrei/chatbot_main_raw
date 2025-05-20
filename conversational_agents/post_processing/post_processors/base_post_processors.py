from abc import ABC, abstractmethod
from data_models.data_models import AgentState, LLMAnswer

class BasePostProcessor(ABC):

    @abstractmethod
    def invoke(agent_state: AgentState, llm_answer: LLMAnswer) -> LLMAnswer:
        pass 