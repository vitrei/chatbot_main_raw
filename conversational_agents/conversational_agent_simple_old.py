from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import json
from dataclasses import asdict

from conversational_agents.agent_logic.base_agent_action import BaseAgentAction
from conversational_agents.agent_logic.base_decision_agent import BaseDecisionAgent
from conversational_agents.agent_logic.base_guiding_instructions import BaseGuidingInstructions
from conversational_agents.base_conversational_agent import ConversationalAgent
from conversational_agents.post_processing.post_processing_pipeline import PostProcessingPipeline
from conversational_agents.pre_processing.pre_processing_pipeline import PreProcessingPipeline
from data_models.data_models import AgentState, LLMAnswer, NextActionDecision, NextActionDecisionType
from large_language_models.llm_factory import llm_factory

class ConversationalAgentSimple(ConversationalAgent):

    def __init__(self, user_id:str, prompts:str, decision_agent:BaseDecisionAgent, agent_logic:BaseAgentAction, guiding_instructions:BaseGuidingInstructions, post_processing_pipeline: PostProcessingPipeline, pre_processing_pipeline: PreProcessingPipeline = None):
        super().__init__()

        self.state = AgentState(
            instruction=None,
            chat_history = {},
            user_id = user_id,
            conversation_turn_counter = 0,
            prompts = prompts
        )

        self.decision_agent = decision_agent
        self.agent_logic = agent_logic
        self.guiding_instructions = guiding_instructions
        self.postprocessing = post_processing_pipeline

        self.preprocessing = pre_processing_pipeline

        self.model_config = {"configurable": {"session_id": self.state.user_id}}

        system_prompt = " ".join(self.state.prompts['system_prompt'])

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        llm = llm_factory.get_llm()

        chain = prompt | llm

        self.chat_chain = RunnableWithMessageHistory(
            chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.state.chat_history:
            self.state.chat_history[session_id] = InMemoryChatMessageHistory()
        return self.state.chat_history[session_id]
    
    async def proactive_instruct(self):
        proactive_prompt = self.state.prompts['proactive_prompt']

        llm_answer_text = ""
        async for chunk in self.chat_chain.astream({"input": proactive_prompt}, config=self.model_config):
            llm_answer_text += chunk.content

        llm_answer = LLMAnswer(
            content=llm_answer_text
        )

        if isinstance(llm_answer, LLMAnswer):
            llm_answer = asdict(llm_answer)  
        return llm_answer

    async def proactive_stream(self): 
        proactive_prompt = self.state.prompts['proactive_prompt']
        
        async for chunk in self.chat_chain.astream({"input": proactive_prompt}, config=self.model_config):
            yield chunk.content

    async def instruct(self, instruction: str):
        self.state.instruction = instruction
        
        # print(f"🔍 DEBUG ConversationalAgent BEFORE pre-processing:")
        # print(f"   - self.state type: {type(self.state)}")
        # print(f"   - self.state id: {id(self.state)}")
        # print(f"   - self.state has user_profile: {hasattr(self.state, 'user_profile')}")

        if self.preprocessing != None:
            print(f"DEBUG: Running pre-processing...")
            self.state = self.preprocessing.invoke(self.state)
            # print(f"DEBUG ConversationalAgent AFTER pre-processing:")
            # print(f"   - self.state type: {type(self.state)}")
            # print(f"   - self.state id: {id(self.state)}")
            # print(f"   - self.state has user_profile: {hasattr(self.state, 'user_profile')}")
            if hasattr(self.state, 'user_profile'):
                print(f"   - self.state.user_profile: {self.state.user_profile}")
        else:
            print(f"DEBUG: No pre-processing pipeline!")

        print(f"DEBUG: Calling decision agent with state id: {id(self.state)}")
        next_action = self.decision_agent.next_action(agent_state=self.state)

        if next_action.type == NextActionDecisionType.PROMPT_ADAPTION: 
            pass

        elif next_action.type == NextActionDecisionType.GUIDING_INSTRUCTIONS: 
            self.state = self.guiding_instructions.add_guiding_instructions(next_action, self.state)

        elif next_action.type == NextActionDecisionType.ACTION:
            llm_answer = self.agent_logic.invoke(next_action, self.state)

        if self.generate_answer(next_action):
            llm_answer_text = ""
            async for chunk in self.chat_chain.astream({"input": self.state.instruction}, config=self.model_config):
                llm_answer_text += chunk.content
            llm_answer = LLMAnswer(
                content=llm_answer_text
            )            

        if self.postprocessing != None:
            llm_answer = self.postprocessing.invoke(self.state, llm_answer) 

        self.state.conversation_turn_counter += 1 
        
        if isinstance(llm_answer, LLMAnswer):
            llm_answer = asdict(llm_answer)  
        return llm_answer

    async def stream(self, instruction: str):   
        self.state.instruction = instruction

        next_action = self.decision_agent.next_action(self.state)

        if next_action.type == NextActionDecisionType.PROMPT_ADAPTION: 
            pass

        elif next_action.type == NextActionDecisionType.GUIDING_INSTRUCTIONS:
            self.state = self.guiding_instructions.add_guiding_instructions(next_action, self.state)

        elif next_action.type == NextActionDecisionType.ACTION:
            llm_answer = self.agent_logic.invoke(next_action, self.state)

        self.state.conversation_turn_counter += 1
        
        if self.generate_answer(next_action):           
            async for chunk in self.chat_chain.astream({"input": self.state.instruction}, config=self.model_config):
                yield chunk.content
        else:
            if isinstance(llm_answer, LLMAnswer):
                yield json.dumps(asdict(llm_answer))  
            else:
                yield str(llm_answer)

    def generate_answer(self, next_action:NextActionDecision):
        return next_action.type in [NextActionDecisionType.PROMPT_ADAPTION, NextActionDecisionType.GENERATE_ANSWER, NextActionDecisionType.GUIDING_INSTRUCTIONS]