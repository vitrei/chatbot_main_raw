from typing import List
from conversational_agents.pre_processing.pre_processors.base_pre_processors import BasePreProcessor
from data_models.data_models import AgentState

class PreProcessingPipeline():

    def __init__(self, pre_processors: List[BasePreProcessor]):
        print("PreProcessing Pipeline initialized with processors:", pre_processors)
        self.pre_processors = pre_processors

    def invoke(self, agent_state: AgentState) -> AgentState:
        for pre_processor in self.pre_processors:
            agent_state = pre_processor.invoke(agent_state)
        return agent_state
