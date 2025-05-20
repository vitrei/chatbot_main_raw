from abc import ABC, abstractmethod

from data_models.data_models import AgentState, NextActionDecision

class BaseGuidingInstructions(ABC):

    @abstractmethod
    def add_guiding_instructions(next_action:NextActionDecision, agent_state: AgentState) -> AgentState:
        pass


class GuidingInstructions(BaseGuidingInstructions):
    
    def __init__(self):
        super().__init__()

    def add_guiding_instructions(self, next_action:NextActionDecision, agent_state: AgentState) -> AgentState:
        gi = agent_state.prompts['guiding_instructions']

        guiding_instruction_name = next_action.action

        if guiding_instruction_name in gi:
            agent_state.instruction += " " + gi[guiding_instruction_name]

        return agent_state