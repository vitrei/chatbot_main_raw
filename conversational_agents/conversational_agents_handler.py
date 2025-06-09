from config import config
from conversational_agents.agent_logic.base_agent_action import AgentAction
from conversational_agents.agent_logic.base_conversational_agent_action_collection import BaseConversationalAgentActionsCollection
from conversational_agents.agent_logic.base_decision_agent import BaseDecisionAgent
from conversational_agents.agent_logic.base_guiding_instructions import GuidingInstructions
from conversational_agents.conversational_agent_rag import ConversationalAgentRAG
from conversational_agents.conversational_agent_simple import ConversationalAgentSimple
from conversational_agents.post_processing.post_processing_pipeline import PostProcessingPipeline
from conversational_agents.pre_processing.pre_processing_pipeline import PreProcessingPipeline
from prompts.prompt_loader import prompt_loader

class ConversationalAgentsHandler():
    
    def __init__(self, agent_logic_actions:BaseConversationalAgentActionsCollection, post_processing_pipeline: PostProcessingPipeline, pre_processing_pipeline: PreProcessingPipeline = None):
        self.conversational_agents = {}
        self.conversational_agents_type = config.get('conversational_agent', 'type')
        if agent_logic_actions != None:
            self.agent_logic_actions = agent_logic_actions.get_actions()
        else: 
            self.agent_logic_actions = {}
        self.post_precessing_pipeline = post_processing_pipeline
        self.pre_processing_pipeline = pre_processing_pipeline

    def initialize_by_user_id(self, user_id: str, decision_agent:BaseDecisionAgent):
        if user_id in self.conversational_agents:
           self.delete_by_user_id(user_id=user_id)
        return self.get_by_user_id(user_id=user_id, decision_agent=decision_agent)

    def get_by_user_id(self, user_id: str, decision_agent:BaseDecisionAgent):
        if user_id in self.conversational_agents:
            return self.conversational_agents[user_id]
        else: 
            guiding_instruction = GuidingInstructions()     
            agent_logic = AgentAction(actions=self.agent_logic_actions)            
                          
            if self.conversational_agents_type == 'simple':                
                prompts = prompt_loader.get_all_prompts()
                new_ca = ConversationalAgentSimple(
                    user_id=user_id, 
                    prompts=prompts, 
                    decision_agent=decision_agent, 
                    agent_logic=agent_logic, 
                    guiding_instructions=guiding_instruction, 
                    post_processing_pipeline=self.post_precessing_pipeline,
                    pre_processing_pipeline=self.pre_processing_pipeline)
                self.conversational_agents[user_id] = new_ca

            elif self.conversational_agents_type == 'rag':
                prompts = prompt_loader.get_all_prompts()               
                new_ca = ConversationalAgentRAG(
                    user_id=user_id, 
                    prompts=prompts, 
                    decision_agent=decision_agent, 
                    agent_logic=agent_logic, 
                    guiding_instructions=guiding_instruction, 
                    post_processing_pipeline=self.post_precessing_pipeline,
                    pre_processing_pipeline=self.pre_processing_pipeline)
                self.conversational_agents[user_id] = new_ca  
                
            return self.conversational_agents[user_id]

    def delete_by_user_id(self, user_id: str):
        if user_id in self.conversational_agents:
            del self.conversational_agents[user_id]


