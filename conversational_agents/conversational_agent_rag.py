from dataclasses import asdict
import json
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain


from conversational_agents.agent_logic.base_agent_action import BaseAgentAction
from conversational_agents.agent_logic.base_decision_agent import BaseDecisionAgent
from conversational_agents.agent_logic.base_guiding_instructions import BaseGuidingInstructions
from conversational_agents.base_conversational_agent import ConversationalAgent
from conversational_agents.post_processing.post_processing_pipeline import PostProcessingPipeline
from conversational_agents.rag_retrievers.rag_retriever_factory import RAGRetrieverFactory
from data_models.data_models import AgentState, LLMAnswer, NextActionDecision, NextActionDecisionType, RAGDocument
from large_language_models.llm_factory import llm_factory

class ConversationalAgentRAG(ConversationalAgent):

    def __init__(self, user_id:str, prompts:str, decision_agent:BaseDecisionAgent, agent_logic:BaseAgentAction, guiding_instructions:BaseGuidingInstructions, post_processing_pipeline: PostProcessingPipeline):
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

        self.model_config = {"configurable": {"session_id": self.state.user_id}}

        system_prompt = " ".join(self.state.prompts['system_prompt'])
        rag_prompt = self.state.prompts['rag_prompt']
        retriever_prompt = self.state.prompts['retriever_prompt']
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", retriever_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )       

        retriever_factory = RAGRetrieverFactory()
        self.retriever = retriever_factory.get_retrievers()                

        llm = llm_factory.get_llm()

        history_aware_retriever = create_history_aware_retriever(
            llm, self.retriever, contextualize_q_prompt
        )    

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt + rag_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        documents_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, documents_chain)

        self.chat_chain = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
    
    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.state.chat_history:
            self.state.chat_history[session_id] = InMemoryChatMessageHistory()
        return self.state.chat_history[session_id]
    
    async def proactive_instruct(self):

        proactive_prompt = self.state.prompts['proactive_prompt']

        llm_answer_rag_result = self.chat_chain.invoke({"input": proactive_prompt}, config=self.model_config)            
        answer = llm_answer_rag_result['answer']

        llm_answer = LLMAnswer(
            content=answer, 
            rag_context = None,
            payload=None 
        )
        
        if isinstance(llm_answer, LLMAnswer):
            llm_answer = asdict(llm_answer) 

        return llm_answer


    async def proactive_stream(self):
        proactive_prompt = self.state.prompts['proactive_prompt']

        llm_answer_rag_result = self.chat_chain.invoke({"input": proactive_prompt}, config=self.model_config)            
        answer = llm_answer_rag_result['answer']

        llm_answer = LLMAnswer(
            content=answer, 
            rag_context = None,
            payload=None 
        )
        
        if isinstance(llm_answer, LLMAnswer):
            yield json.dumps(asdict(llm_answer))  
        else:
            yield str(llm_answer)

    async def instruct(self, instruction: str):
        self.state.instruction = instruction
        print("state", self.state)
        print()
        next_action = self.decision_agent.next_action(self.state)
        print("next_action", asdict(next_action))
        print()

        llm_answer = LLMAnswer(
            content=None, 
            rag_context = None,
            payload=None 
        )

        if next_action.type == NextActionDecisionType.PROMPT_ADAPTION: 
            pass # changing system prompt etc.

        elif next_action.type == NextActionDecisionType.GUIDING_INSTRUCTIONS:
            self.state = self.guiding_instructions.add_guiding_instructions(next_action, self.state)

        elif next_action.type == NextActionDecisionType.ACTION:
            llm_answer = self.agent_logic.invoke(next_action, self.state)
        
        if self.generate_answer(next_action):
                        
            llm_answer_rag_result = self.chat_chain.invoke({"input": self.state.instruction}, config=self.model_config)
            
            answer = llm_answer_rag_result['answer']
            context = llm_answer_rag_result['context']

            rag_documents = []
            for doc in context:
                rag_document = RAGDocument(
                    content = doc.page_content,
                    metadata = doc.metadata
                )                
                rag_documents.append(rag_document)
           
            llm_answer.content = answer
            llm_answer.rag_context = rag_documents

        if self.postprocessing != None:
            llm_answer = self.postprocessing.invoke(self.state, llm_answer)  
        
        self.state.conversation_turn_counter += 1

        if isinstance(llm_answer, LLMAnswer):
            llm_answer = asdict(llm_answer)  
        return llm_answer


    async def stream(self, instruction: str):
        self.state.instruction = instruction

        next_action = self.decision_agent.next_action(self.state)
        print("next_action", next_action)

        llm_answer = LLMAnswer(
            content=None, 
            rag_context = None,
            payload=None 
        )

        if next_action.type == NextActionDecisionType.PROMPT_ADAPTION: 
            pass # changing system prompt etc.

        elif next_action.type == NextActionDecisionType.GUIDING_INSTRUCTIONS:
            self.state = self.guiding_instructions.add_guiding_instructions(next_action, self.state)

        elif next_action.type == NextActionDecisionType.ACTION:
            llm_answer = self.agent_logic.invoke(next_action, self.state)
        
        if self.generate_answer(next_action):

            llm_answer_rag_result = self.chat_chain.invoke({"input": self.state.instruction}, config=self.model_config)
           
            answer = llm_answer_rag_result['answer']
            context = llm_answer_rag_result['context']

            rag_documents = []
            for doc in context:
                rag_document = RAGDocument(
                    content = doc.page_content,
                    metadata = doc.metadata
                )                
                rag_documents.append(rag_document)
           
            llm_answer.content = answer
            llm_answer.rag_context = rag_documents

        if self.postprocessing != None:
            llm_answer = self.postprocessing.invoke(self.state, llm_answer) 
        
        self.state.conversation_turn_counter += 1
        
        if isinstance(llm_answer, LLMAnswer):
            yield json.dumps(asdict(llm_answer))  
        else:
            yield str(llm_answer)


    def generate_answer(self, next_action:NextActionDecision):
        return next_action.type in [NextActionDecisionType.PROMPT_ADAPTION, NextActionDecisionType.GENERATE_ANSWER, NextActionDecisionType.GUIDING_INSTRUCTIONS]