from typing import List
from conversational_agents.post_processing.post_processors.base_post_processors import BasePostProcessor
from data_models.data_models import AgentState, LLMAnswer


class PostProcessingPipeline():

    def __init__(self, post_processors: List[BasePostProcessor]):
        print(post_processors)
        self.post_processors = post_processors

    def invoke(self, agent_state: AgentState, llm_answer: LLMAnswer):
        for post_processor in self.post_processors:
            llm_answer  = post_processor.invoke(llm_answer)
        return llm_answer