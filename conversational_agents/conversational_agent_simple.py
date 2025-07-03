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
                ("system", system_prompt + "\n\n{user_profile_context}"),
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

    def get_current_state_from_state_machine(self):
        """Extract current state from state machine"""
        if hasattr(self.state, 'state_machine') and self.state.state_machine:
            return self.state.state_machine.get_current_state()
        return "init_greeting"  # fallback to initial state

    # DEPRECATED: Replaced by build_dynamic_prompt
    # def get_current_system_prompt(self): ...

    def get_state_examples(self, current_state):
        """Get few-shot examples for current state (prioritize injected from state machine)"""
        try:
            # Use injected state machine examples if available
            if hasattr(self.state, 'current_state_examples') and self.state.current_state_examples:
                examples = self.state.current_state_examples
                print(f"🔄 Using injected state machine examples for {current_state}")
            else:
                # Fallback to examples from prompts_fake_news.json (legacy)
                state_examples = self.state.prompts.get('state_examples', {})
                examples = state_examples.get(current_state, [])
                print(f"⚠️ Using legacy examples for {current_state}")
            
            # Convert JSON format to ChatPromptTemplate format
            formatted_examples = []
            for example in examples:
                role = example.get('role', 'human')
                content = example.get('content', '')
                formatted_examples.append((role, content))
            
            # print(f"🎭 LOADED {len(formatted_examples)} EXAMPLES for state: {current_state}")
            return formatted_examples
        except Exception as e:
            print(f"❌ ERROR loading examples for {current_state}: {e}")
            return []

    def build_dynamic_prompt(self, current_state, stage_context=None):
        """Master prompt builder - single source of truth for all conversation prompts"""
        try:
            # print(f"🎨 BUILDING DYNAMIC PROMPT for state: {current_state}")
            
            # === 1. BASE SYSTEM PROMPT (from prompts_fake_news.json) ===
            base_system_prompt = " ".join(self.state.prompts['system_prompt'])
            # print(f"  ✅ Base system prompt loaded")
            
            # === 2. STATE-SPECIFIC INSTRUCTIONS (from state_machine.json) ===
            state_instructions = []
            if hasattr(self.state, 'current_state_prompts') and self.state.current_state_prompts:
                # Use injected state machine prompts (preferred)
                state_instructions = self.state.current_state_prompts.copy()
                # print(f"  ✅ State instructions from state machine: {len(state_instructions)} items")
            else:
                print(f"  ⚠️ No state machine prompts available for {current_state}")
            
            # Override with dynamic prompt for stage_selection if available
            if stage_context and current_state == 'stage_selection' and stage_context.get('dynamic_stage_prompt'):
                dynamic_prompt = stage_context['dynamic_stage_prompt']
                state_instructions = [dynamic_prompt]  # Replace with dynamic prompt
                # print(f"  🎯 Using dynamic stage selection prompt")
            elif stage_context and current_state == 'stage_selection' and stage_context.get('unavailable_stage_requested'):
                print(f"  🚫 Stage selection with unavailable stage request")
            
            # === 3. BEHAVIORAL GUIDANCE (from guiding instructions) ===
            behavioral_guidance = "Natürlich und locker sprechen."
            if hasattr(self.state, 'current_guiding_instruction'):
                # Extract behavioral part only
                full_instruction = self.state.current_guiding_instruction
                if "VERHALTEN:" in full_instruction:
                    behavioral_part = full_instruction.split("INHALT/PHASE:")[0].replace("VERHALTEN:", "").strip()
                    if behavioral_part:
                        behavioral_guidance = behavioral_part
                # print(f"  ✅ Behavioral guidance: {behavioral_guidance[:50]}...")
            
            # === 4. FEW-SHOT EXAMPLES (from state_machine.json) ===
            examples = self.get_state_examples(current_state)
            # print(f"  ✅ Examples loaded: {len(examples)} examples")
            
            # === BUILD COMPREHENSIVE SYSTEM PROMPT ===
            system_prompt_components = [
                "=== GRUNDREGELN ===",
                base_system_prompt,
                "",
                f"=== AKTUELLE SZENE: {current_state.upper()} ===",
                "Du befindest dich gerade in dieser spezifischen Gesprächsphase:"
            ]
            
            # Add state-specific instructions
            if state_instructions:
                system_prompt_components.append("AUFGABEN FÜR DIESE SZENE:")
                for i, instruction in enumerate(state_instructions, 1):
                    system_prompt_components.append(f"{i}. {instruction}")
            else:
                system_prompt_components.append("[Keine spezifischen Anweisungen für diese Szene]")
            
            system_prompt_components.extend([
                "",
                "=== VERHALTEN UND STIL ===",
                behavioral_guidance,
                "",
                "=== PRIORITÄTEN ===",
                "1. Befolge deine Szenen-Aufgaben",
                "2. Reagiere angemessen auf User-Input", 
                "3. Halte den vorgegebenen Stil bei",
                "4. Bewege das Gespräch zielgerichtet voran"
            ])
            
            system_prompt = "\n".join(system_prompt_components)
            
            # === BUILD COMPLETE MESSAGE CHAIN ===
            messages = [("system", system_prompt)]
            
            # Add few-shot examples for reinforcement
            if examples:
                messages.extend(examples)
                # print(f"  ✅ Added {len(examples)} few-shot examples")
            
            # Add chat history and current input placeholders
            messages.extend([
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])
            
            # print(f"  ✅ Dynamic prompt built successfully with {len(messages)} components")
            return ChatPromptTemplate.from_messages(messages)
            
        except Exception as e:
            print(f"❌ ERROR building dynamic prompt: {e}")
            # Minimal fallback
            fallback_prompt = " ".join(self.state.prompts['system_prompt'])
            return ChatPromptTemplate.from_messages([
                ("system", fallback_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])
    
    # TODO: Will be integrated into build_dynamic_prompt later
    # def format_user_profile_for_llm(self): ...

    # TODO: Could be re-implemented for quality assurance later
    
    async def proactive_instruct(self):
        proactive_prompt = self.state.prompts['proactive_prompt']

        # TODO: User profile context will be integrated into build_dynamic_prompt later
        user_profile_context = ""  # Removed for now

        llm_answer_text = ""
        async for chunk in self.chat_chain.astream({
            "input": proactive_prompt,
            "user_profile_context": user_profile_context
        }, config=self.model_config):
            llm_answer_text += chunk.content

        llm_answer = LLMAnswer(content=llm_answer_text)

        if isinstance(llm_answer, LLMAnswer):
            llm_answer = asdict(llm_answer)  
        return llm_answer


    async def instruct(self, instruction: str):
        self.state.instruction = instruction
        
        if self.preprocessing != None:
            print(f"DEBUG: Running pre-processing...")
            self.state = self.preprocessing.invoke(self.state)
        else:
            print(f"DEBUG: No pre-processing pipeline!")

        print(f"DEBUG: Calling decision agent with state id: {id(self.state)}")
        next_action = self.decision_agent.next_action(agent_state=self.state)

        if next_action.type == NextActionDecisionType.PROMPT_ADAPTION:
            # Inject state machine context into agent state
            if next_action.payload:
                # print(f"🔄 PROMPT_ADAPTION: Injecting state machine context for {next_action.payload.get('current_state')}")
                
                # Store state machine prompts and examples in agent state
                if 'state_system_prompts' in next_action.payload:
                    self.state.current_state_prompts = next_action.payload['state_system_prompts']
                if 'state_examples' in next_action.payload:
                    self.state.current_state_examples = next_action.payload['state_examples']
                
                # Apply the guiding instruction after injecting state context
                guiding_instruction_name = next_action.payload.get('guiding_instruction', 'general_guidance')
                
                # Create a secondary NextActionDecision for guiding instructions
                guiding_decision = NextActionDecision(
                    type=NextActionDecisionType.GUIDING_INSTRUCTIONS,
                    action=guiding_instruction_name,
                    payload=next_action.payload
                )
                
                self.state = self.guiding_instructions.add_guiding_instructions(guiding_decision, self.state)
                self.state.current_guiding_instruction_name = guiding_instruction_name

        elif next_action.type == NextActionDecisionType.GUIDING_INSTRUCTIONS: 
            self.state = self.guiding_instructions.add_guiding_instructions(next_action, self.state)
            # Store current guiding instruction name for behavioral reference
            self.state.current_guiding_instruction_name = next_action.action

        elif next_action.type == NextActionDecisionType.ACTION:
            llm_answer = self.agent_logic.invoke(next_action, self.state)

        if self.generate_answer(next_action):
            # Get REAL current state from state machine (not behavioral instruction)
            current_state = self.get_current_state_from_state_machine()
            # print(f"🎯 ACTUAL STATE MACHINE STATE: {current_state}")
            
            # Get full state machine context (includes dynamic prompts and stage info) - MOVED EARLIER
            stage_context = {}
            if hasattr(self.state, 'state_machine') and self.state.state_machine:
                available_transitions = self.state.state_machine.get_available_transitions()
                stage_context = self.state.state_machine.get_state_context_for_decision_agent(self.state.conversation_turn_counter)
                # print(f"🔄 AVAILABLE TRANSITIONS: {[t['trigger'] for t in available_transitions]}")
                
                # Check for unavailable stage requests in stage_selection
                if current_state == 'stage_selection' and stage_context.get('unavailable_stage_requested'):
                    print(f"🚫 UNAVAILABLE STAGE REQUESTED: {stage_context['unavailable_stage_requested']}")
                elif current_state == 'stage_selection' and stage_context.get('dynamic_stage_prompt'):
                    print(f"📋 DYNAMIC STAGE PROMPT AVAILABLE")
            
            # Build dynamic prompt with state-specific content and examples
            dynamic_prompt = self.build_dynamic_prompt(current_state, stage_context)
            
            llm = llm_factory.get_llm()
            chain = dynamic_prompt | llm

            chat_chain = RunnableWithMessageHistory(
                chain,
                self.get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history"
            )
            
            # TODO: User profile context will be integrated into build_dynamic_prompt later
            user_profile_context = ""  # Removed for now
            
            # print(f"🎯 BEHAVIORAL INSTRUCTION: {getattr(self.state, 'current_guiding_instruction_name', 'None')}")
            # print(f"💭 COMBINED GUIDANCE: {getattr(self.state, 'current_guiding_instruction', 'None')[:100] if hasattr(self.state, 'current_guiding_instruction') else 'None'}...")
            # print(f"DEBUG: Using state-specific prompt and examples for state machine state: {current_state}")
            
            # Debug: Show what exact instructions the LLM gets (use injected prompts)
            if hasattr(self.state, 'current_state_prompts') and self.state.current_state_prompts:
                state_instructions = self.state.current_state_prompts
                # print(f"🎭 EXACT STATE INSTRUCTIONS SENT TO LLM (from state machine):")
            else:
                state_prompts = self.state.prompts.get('state_system_prompts', {})
                state_instructions = state_prompts.get(current_state, [])
                print(f"🎭 EXACT STATE INSTRUCTIONS SENT TO LLM (legacy):")
            
            for i, instruction in enumerate(state_instructions, 1):
                print(f"   {i}. {instruction}")
            
            llm_answer_text = ""
            async for chunk in chat_chain.astream({
                "input": self.state.instruction
            }, config=self.model_config):
                llm_answer_text += chunk.content
                
            # print(f"🤖 LLM RESPONSE: {llm_answer_text[:100]}...")
            # print(f"🔍 RESPONSE ANALYSIS: Does it follow state instructions? {self.analyze_response_compliance(current_state, llm_answer_text)}")
            llm_answer = LLMAnswer(content=llm_answer_text)       

        if self.postprocessing != None:
            llm_answer = self.postprocessing.invoke(self.state, llm_answer) 

        self.state.conversation_turn_counter += 1 
        
        if isinstance(llm_answer, LLMAnswer):
            llm_answer = asdict(llm_answer)  
        return llm_answer

    def generate_answer(self, next_action:NextActionDecision):
        return next_action.type in [NextActionDecisionType.PROMPT_ADAPTION, NextActionDecisionType.GENERATE_ANSWER, NextActionDecisionType.GUIDING_INSTRUCTIONS]

    async def proactive_stream(self): 
        proactive_prompt = self.state.prompts['proactive_prompt']
        
        # TODO: User profile context will be integrated into build_dynamic_prompt later
        user_profile_context = ""  # Removed for now
        
        async for chunk in self.chat_chain.astream({
            "input": proactive_prompt,
            "user_profile_context": user_profile_context
        }, config=self.model_config):
            yield chunk.content
    
    async def stream(self, instruction: str):   
        self.state.instruction = instruction

        if self.preprocessing != None:
            self.state = self.preprocessing.invoke(self.state)

        next_action = self.decision_agent.next_action(self.state)

        if next_action.type == NextActionDecisionType.PROMPT_ADAPTION: 
            pass

        elif next_action.type == NextActionDecisionType.GUIDING_INSTRUCTIONS:
            self.state = self.guiding_instructions.add_guiding_instructions(next_action, self.state)

        elif next_action.type == NextActionDecisionType.ACTION:
            llm_answer = self.agent_logic.invoke(next_action, self.state)

        self.state.conversation_turn_counter += 1
        
        if self.generate_answer(next_action):
            # Use real state machine state for streaming too
            current_state = self.get_current_state_from_state_machine()
            
            # Get stage context for streaming too
            stage_context = {}
            if hasattr(self.state, 'state_machine') and self.state.state_machine:
                stage_context = self.state.state_machine.get_state_context_for_decision_agent(self.state.conversation_turn_counter)
            
            dynamic_prompt = self.build_dynamic_prompt(current_state, stage_context)
            
            llm = llm_factory.get_llm()
            chain = dynamic_prompt | llm

            chat_chain = RunnableWithMessageHistory(
                chain,
                self.get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history"
            )
            
            # TODO: User profile context will be integrated into build_dynamic_prompt later
            user_profile_context = ""  # Removed for now
            
            async for chunk in chat_chain.astream({
                "input": self.state.instruction,
                # "user_profile_context": user_profile_context  # Removed for now
            }, config=self.model_config):
                yield chunk.content
        else:
            if isinstance(llm_answer, LLMAnswer):
                yield json.dumps(asdict(llm_answer))  
            else:
                yield str(llm_answer)