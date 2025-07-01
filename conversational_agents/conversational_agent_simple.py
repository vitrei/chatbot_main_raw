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

    def get_current_system_prompt(self):
        """Build system prompt including current state-specific prompt and guiding instructions"""
        base_system_prompt = " ".join(self.state.prompts['system_prompt'])
        
        # Get current state from state machine (not from guiding instruction)
        current_state = self.get_current_state_from_state_machine()
        
        # Get state-specific system prompt
        state_prompts = self.state.prompts.get('state_system_prompts', {})
        state_specific_prompt = state_prompts.get(current_state, [])
        
        # Get current guiding instruction (behavioral guidance)
        behavioral_instruction = ""
        if hasattr(self.state, 'current_guiding_instruction'):
            behavioral_instruction = self.state.current_guiding_instruction
        
        # Build comprehensive system prompt with STRONG emphasis on state instructions
        prompt_parts = [base_system_prompt]
        
        if state_specific_prompt:
            state_prompt_text = " ".join(state_specific_prompt)
            prompt_parts.append(f"🎭 DREHBUCH/PFLICHTAUFGABE ({current_state}): {state_prompt_text}")
            prompt_parts.append("WICHTIG: Du MUSST diese Drehbuch-Anweisungen befolgen! Ignoriere nicht was der User sagt, aber folge primär dem Drehbuch!")
        
        if behavioral_instruction:
            prompt_parts.append(f"🎪 STIL UND VERHALTEN: {behavioral_instruction}")
        
        # Add stronger instruction hierarchy
        prompt_parts.append("PRIORITÄTEN: 1) Befolge das DREHBUCH, 2) Berücksichtige User-Input, 3) Halte den vorgeschriebenen STIL bei")
        
        combined_prompt = "\n\n".join(prompt_parts)
        return f"{combined_prompt}\n\n{{user_profile_context}}"

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

    def build_dynamic_prompt(self, current_state):
        """Build dynamic prompt with STRONG state-specific enforcement"""
        try:
            # Get state-specific instructions and examples (prioritize injected from state machine)
            if hasattr(self.state, 'current_state_prompts') and self.state.current_state_prompts:
                # Use injected state machine prompts
                state_instructions = self.state.current_state_prompts
                print(f"🔄 Using injected state machine prompts for {current_state}")
            else:
                # Fallback to prompts from prompts_fake_news.json (legacy)
                state_prompts = self.state.prompts.get('state_system_prompts', {})
                state_instructions = state_prompts.get(current_state, [])
                print(f"⚠️ Using legacy prompts for {current_state}")
            examples = self.get_state_examples(current_state)
            
            # Get behavioral guidance
            behavioral_instruction = ""
            if hasattr(self.state, 'current_guiding_instruction'):
                # Extract only the behavioral part (before "INHALT/PHASE:")
                full_instruction = self.state.current_guiding_instruction
                if "VERHALTEN:" in full_instruction:
                    behavioral_part = full_instruction.split("INHALT/PHASE:")[0].replace("VERHALTEN:", "").strip()
                    behavioral_instruction = behavioral_part
            
            # Build DIRECTIVE system prompt
            base_prompt = " ".join(self.state.prompts['system_prompt'])
            
            # Create VERY directive system prompt
            system_prompt_parts = [
                base_prompt,
                "",
                "🎭 DEINE AKTUELLE SZENE/AUFGABE:",
                f"State: {current_state}",
                "Du MUSST folgende Aufgabe erfüllen:",
            ]
            
            if state_instructions:
                for instruction in state_instructions:
                    system_prompt_parts.append(f"- {instruction}")
            
            system_prompt_parts.extend([
                "",
                "🎪 WIE DU DICH VERHALTEN SOLLST:",
                behavioral_instruction if behavioral_instruction else "Natürlich und locker sprechen.",
                "",
                "⚠️ WICHTIG: Du befolgst ZUERST deine Szenen-Aufgabe, dann berücksichtigst du den User-Input!",
                "Ignoriere NICHT was der User sagt, aber deine Haupt-Priorität ist es, deine Szenen-Aufgabe zu erfüllen!",
                "",
                "{user_profile_context}"
            ])
            
            system_prompt = "\n".join(system_prompt_parts)
            
            # Build messages with examples
            messages = [("system", system_prompt)]
            
            # Add few-shot examples to REINFORCE the correct behavior
            if examples:
                messages.extend(examples)
                # print(f"🎭 ADDED {len(examples)} REINFORCEMENT EXAMPLES")
            
            # Add chat history and current input
            messages.extend([
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])
            
            return ChatPromptTemplate.from_messages(messages)
            
        except Exception as e:
            print(f"❌ ERROR building dynamic prompt: {e}")
            # Fallback
            return ChatPromptTemplate.from_messages([
                ("system", self.get_current_system_prompt()),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])
    
    def format_user_profile_for_llm(self) -> str:
        """
        Format user profile for inclusion in LLM context - FIXED VERSION
        """
        if not hasattr(self.state, 'user_profile') or not self.state.user_profile:
            return ""
        
        profile = self.state.user_profile
        
        profile_info = []
        instructions = []
        
        # Safe age handling with type conversion
        age = profile.get('age')
        if age:
            try:
                age_int = int(age)  # Convert to int if it's a string
                profile_info.append(f"{age_int}J")
            except (ValueError, TypeError):
                # If conversion fails, just use the original value as string
                profile_info.append(f"{age}J")
                age_int = None  # Set to None for later comparisons
        else:
            age_int = None
            
        if profile.get('school_type'):
            profile_info.append(f"{profile['school_type']}")
        if profile.get('region'):
            profile_info.append(f"{profile['region']}")
            
        fake_news_skill = profile.get('fake_news_skill')
        if fake_news_skill:
            if fake_news_skill == 'master':
                profile_info.append("FN:Experte")
                instructions.append("kritisch fragen")
            elif fake_news_skill == 'low':
                profile_info.append("FN:Anfänger")
                instructions.append("einfach erklären")
            else:
                profile_info.append(f"FN:{fake_news_skill}")
        
        if profile.get('attention_span') == 'short':
            profile_info.append("Aufm:kurz")
            instructions.append("max 150 Zeichen")
        
        current_mood = profile.get('current_mood')
        if current_mood == 'mad':
            profile_info.append("Stimmung:schlecht")
            instructions.append("einfühlsam sein")
        elif current_mood == 'enthusiastic':
            profile_info.append("Stimmung:motiviert")
        
        if profile.get('interaction_style'):
            style = profile['interaction_style']
            if style == 'direct':
                profile_info.append("Stil:direkt")
                instructions.append("klar sprechen")
            elif style == 'gentle':
                profile_info.append("Stil:sanft")
                instructions.append("vorsichtig sein")
        
        if profile.get('interests'):
            interests = profile['interests'][:2]  # Nur erste 2
            profile_info.append(f"Interesse:{','.join(interests)}")
        
        # Safe age comparison using converted integer
        if age_int is not None and age_int < 16:
            instructions.append("jugendlich sprechen")
            
        if fake_news_skill == 'master':
            instructions.append("nicht leicht zufriedenstellen")
        elif fake_news_skill == 'low':
            instructions.append("geduldig bleiben")
            
        if current_mood == 'mad':
            instructions.append("Konfrontation vermeiden")
            
        if profile.get('attention_span') == 'short':
            instructions.append("direkt zum Punkt")
        
        output_parts = []
        
        if profile_info:
            output_parts.append(f"User: {' | '.join(profile_info)}")
        
        if instructions:
            output_parts.append(f"Anpassungen: {', '.join(instructions)}")
        
        return " || ".join(output_parts) if output_parts else ""

    # def analyze_response_compliance(self, current_state, response_text):
    #     """Analyze if the response follows state instructions"""
    #     state_keywords = {
    #         'engagement_hook': ['video', 'sieht', 'ähnlich', 'komisch', 'weird'],
    #         'stimulus_present': ['tanzen', 'schule', 'ernst', 'untypisch', 'peinlich', 'überraschend'],
    #         'reaction_wait': ['gefühlt', 'weird', 'reaktion'],
    #         'explore_path': ['glaubwürdig', 'dagegen', 'sprechen'],
    #         'comfort_user': ['entspann', 'video gibt es nicht', 'demonstration'],
    #         'confirm_skepticism': ['gut beobachtet', 'skeptisch', 'wichtig']
    #     }
        
    #     keywords = state_keywords.get(current_state, [])
    #     matches = sum(1 for keyword in keywords if keyword.lower() in response_text.lower())
    #     compliance_score = matches / len(keywords) if keywords else 0
        
    #     return f"{compliance_score:.1%} (matched {matches}/{len(keywords)} keywords)"
    
    async def proactive_instruct(self):
        proactive_prompt = self.state.prompts['proactive_prompt']

        user_profile_context = self.format_user_profile_for_llm()

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
                print(f"🔄 PROMPT_ADAPTION: Injecting state machine context for {next_action.payload.get('current_state')}")
                
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
            
            # Build dynamic prompt with state-specific content and examples
            dynamic_prompt = self.build_dynamic_prompt(current_state)
            
            llm = llm_factory.get_llm()
            chain = dynamic_prompt | llm

            chat_chain = RunnableWithMessageHistory(
                chain,
                self.get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history"
            )
            
            user_profile_context = self.format_user_profile_for_llm()
            
            # Debug output showing state machine context
            if hasattr(self.state, 'state_machine') and self.state.state_machine:
                available_transitions = self.state.state_machine.get_available_transitions()
                # print(f"🎰 STATE MACHINE STATE: {current_state}")
                print(f"🔄 AVAILABLE TRANSITIONS: {[t['trigger'] for t in available_transitions]}")
            
            # print(f"🎯 BEHAVIORAL INSTRUCTION: {getattr(self.state, 'current_guiding_instruction_name', 'None')}")
            # print(f"💭 COMBINED GUIDANCE: {getattr(self.state, 'current_guiding_instruction', 'None')[:100] if hasattr(self.state, 'current_guiding_instruction') else 'None'}...")
            # print(f"DEBUG: Using state-specific prompt and examples for state machine state: {current_state}")
            
            # Debug: Show what exact instructions the LLM gets (use injected prompts)
            if hasattr(self.state, 'current_state_prompts') and self.state.current_state_prompts:
                state_instructions = self.state.current_state_prompts
                print(f"🎭 EXACT STATE INSTRUCTIONS SENT TO LLM (from state machine):")
            else:
                state_prompts = self.state.prompts.get('state_system_prompts', {})
                state_instructions = state_prompts.get(current_state, [])
                print(f"🎭 EXACT STATE INSTRUCTIONS SENT TO LLM (legacy):")
            
            for i, instruction in enumerate(state_instructions, 1):
                print(f"   {i}. {instruction}")
            
            llm_answer_text = ""
            async for chunk in chat_chain.astream({
                "input": self.state.instruction,
                "user_profile_context": user_profile_context
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
        
        user_profile_context = self.format_user_profile_for_llm()
        
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
            dynamic_prompt = self.build_dynamic_prompt(current_state)
            
            llm = llm_factory.get_llm()
            chain = dynamic_prompt | llm

            chat_chain = RunnableWithMessageHistory(
                chain,
                self.get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history"
            )
            
            user_profile_context = self.format_user_profile_for_llm()
            
            async for chunk in chat_chain.astream({
                "input": self.state.instruction,
                "user_profile_context": user_profile_context
            }, config=self.model_config):
                yield chunk.content
        else:
            if isinstance(llm_answer, LLMAnswer):
                yield json.dumps(asdict(llm_answer))  
            else:
                yield str(llm_answer)