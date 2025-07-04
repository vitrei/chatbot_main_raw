from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import json
import time
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
        
        self._cached_base_prompt = " ".join(prompts['system_prompt'])
        self._cached_examples = {}

        self.decision_agent = decision_agent
        self.agent_logic = agent_logic
        self.guiding_instructions = guiding_instructions
        self.postprocessing = post_processing_pipeline
        self.preprocessing = pre_processing_pipeline

        self.model_config = {"configurable": {"session_id": self.state.user_id}}

        system_prompt = " ".join(self.state.prompts['system_prompt'])
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt+ "\n\n{user_profile_context}"),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # Use agent-specific model from config
        from config import config
        conversational_model = config.get("llm", "conversational_agent_model", fallback=config.get("llm", "model_name"))
        llm = llm_factory.get_llm(conversational_model)
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
        return "init_greeting" 


    def get_state_examples(self, current_state):
        """Get few-shot examples for current state (prioritize injected from state machine)"""
        try:
            # PERFORMANCE: Check cache first
            if current_state in self._cached_examples:
                return self._cached_examples[current_state]
            
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
            
            # PERFORMANCE: Cache the result
            self._cached_examples[current_state] = formatted_examples
            return formatted_examples
        except Exception as e:
            print(f"❌ ERROR loading examples for {current_state}: {e}")
            return []

    def build_dynamic_prompt(self, current_state, stage_context=None):
        """Master prompt builder - single source of truth for all conversation prompts"""
        try:
            print(f"🎨 BUILDING DYNAMIC PROMPT for state: {current_state}")
            
            # === 1. BASE SYSTEM PROMPT (from prompts_fake_news.json) ===
            base_system_prompt = " ".join(self.state.prompts['system_prompt'])
            print(f"  ✅ Base system prompt loaded ({len(base_system_prompt)} chars)")
            
            # === 2. STATE-SPECIFIC INSTRUCTIONS (from state_machine.json) ===
            state_instructions = []
            if hasattr(self.state, 'current_state_prompts') and self.state.current_state_prompts:
                # Use injected state machine prompts (preferred)
                state_instructions = self.state.current_state_prompts.copy()
                print(f"  ✅ State instructions from state machine: {len(state_instructions)} items")
                for i, instruction in enumerate(state_instructions[:3], 1):  # Show first 3
                    print(f"    {i}. {instruction[:80]}...")
            else:
                print(f"  ⚠️ No state machine prompts available for {current_state}")
            
            # Override with dynamic prompt for stage_selection if available
            if stage_context and current_state == 'stage_selection' and stage_context.get('dynamic_stage_prompt'):
                dynamic_prompt = stage_context['dynamic_stage_prompt']
                state_instructions = [dynamic_prompt]  # Replace with dynamic prompt
                print(f"  🎯 Using dynamic stage selection prompt ({len(dynamic_prompt)} chars)")
            elif stage_context and current_state == 'stage_selection' and stage_context.get('unavailable_stage_requested'):
                print(f"  🚫 Stage selection with unavailable stage request")
            
            # === 3. BEHAVIORAL GUIDANCE (from guiding instructions) ===
            # Start with general guidance from prompts_fake_news.json
            general_guidance = self.state.prompts.get('guiding_instructions', {}).get('general_guidance', "Natürlich und locker sprechen.")
            behavioral_guidance = general_guidance
            
            # Override with specific guiding instruction if available
            if hasattr(self.state, 'current_guiding_instruction'):
                # Extract behavioral part only
                full_instruction = self.state.current_guiding_instruction
                if "VERHALTEN:" in full_instruction:
                    behavioral_part = full_instruction.split("INHALT/PHASE:")[0].replace("VERHALTEN:", "").strip()
                    if behavioral_part:
                        behavioral_guidance = behavioral_part
                else:
                    # Use full instruction if no VERHALTEN: split
                    behavioral_guidance = full_instruction
                print(f"  ✅ Using specific behavioral guidance: {behavioral_guidance[:80]}...")
            else:
                print(f"  ✅ Using general behavioral guidance: {behavioral_guidance[:80]}...")
            
            # === 4. FEW-SHOT EXAMPLES (from state_machine.json) ===
            examples = self.get_state_examples(current_state)
            print(f"  ✅ Examples loaded: {len(examples)} examples")
            if examples:
                for i, (role, content) in enumerate(examples[:2], 1):  # Show first 2 examples
                    print(f"    Example {i} ({role}): {content[:60]}...")
            
            # === BUILD COMPREHENSIVE SYSTEM PROMPT ===
            current_stage = self._get_current_stage()
            storyline_context = self._get_simple_storyline_context(current_stage, current_state)
            print(f"  ✅ Stage context: {storyline_context[:80]}...")
            
            system_prompt_components = [
                "=== HAUPTAUFGABE: BLEIBE BEIM DREHBUCH ===",
                "Du bist ein zielorientierter Fake-News-Aufklärungs-Bot mit einem strukturierten Drehbuch.",
                "Deine oberste Priorität ist das STATE-SCRIPT zu befolgen, nicht User-Ablenkungen zu folgen!",
                "",
                "🎭 DU BIST DER REGISSEUR: Du führst das Gespräch, der User ist nicht der Chef!",
                "",
                "=== GRUNDREGELN ===", 
                base_system_prompt,
                "",
                "=== FAKE NEWS BILDUNGSREISE ===",
                storyline_context,
                "",
                f"=== DEIN AKTUELLER AUFTRAG: {current_state.upper()} ===",
                "🎬 DREHBUCH-ANWEISUNGEN FÜR DIESE SZENE:"
            ]
            
            # Add state-specific instructions with emphasis
            if state_instructions:
                for i, instruction in enumerate(state_instructions, 1):
                    system_prompt_components.append(f"   {i}. ✅ {instruction}")
                system_prompt_components.append("")
                system_prompt_components.append("⚠️  WICHTIG: Diese Aufgaben sind dein HAUPTZIEL - nicht die User-Fragen!")
            else:
                system_prompt_components.append("[❌ Keine spezifischen Drehbuch-Anweisungen für diese Szene]")
            
            system_prompt_components.extend([
                "",
                "=== REAKTIONS-STRATEGIE ===",
                "🎯 PRIMÄR: Führe deine STATE-AUFGABEN aus (siehe oben)",
                "👂 SEKUNDÄR: Nimm User-Input zur Kenntnis",
                "🔄 Falls User ablenkt:",
                "   - Kurz anerkennen ('Verstehe...' / 'Das ist interessant, aber...')",
                "   - SOFORT zurück zu deinem State-Auftrag lenken", 
                "   - State-Aufgaben haben immer Vorrang vor User-Wünschen",
                "",
                "=== VERHALTEN UND STIL ===",
                behavioral_guidance,
                "",
                "=== HIERARCHIE (in dieser Reihenfolge) ===",
                "1. 🎬 STATE-DREHBUCH befolgen (oberste Priorität)",
                "2. 📚 Bildungsziel der aktuellen Stage vorantreiben",
                "3. 👂 User-Input kurz würdigen, falls nicht ablenkend",
                "4. 🎭 Vorgegebenen Stil beibehalten",
                "",
                "⚠️  REMEMBER: Du bist der REGISSEUR, nicht der User!"
            ])
            
            system_prompt = "\n".join(system_prompt_components)
            print(f"  📏 Final system prompt: {len(system_prompt)} chars")
            
            # === BUILD COMPLETE MESSAGE CHAIN ===
            messages = [("system", system_prompt)]
            
            # Add few-shot examples for reinforcement
            if examples:
                messages.extend(examples)
                print(f"  ✅ Added {len(examples)} few-shot examples")
            
            # Add chat history and current input placeholders
            messages.extend([
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])
            
            print(f"  ✅ Dynamic prompt built successfully with {len(messages)} total components")
            print(f"  🎭 FINAL PROMPT STRUCTURE:")
            print(f"     - System prompt: {len(system_prompt)} chars")
            print(f"     - Examples: {len(examples)} items")
            print(f"     - Total messages: {len(messages)} components")
            
            # Debug: Show complete final system prompt structure
            print(f"🎯 FINAL SYSTEM PROMPT BREAKDOWN:")
            # print(f"  📏 Total length: {len(system_prompt)} characters")
            print(f"  🔍 First 200 chars: {system_prompt}")
            # print(f"  🔍 First 200 chars: {system_prompt[:200]}...")
            # print(f"  🔍 Last 200 chars: ...{system_prompt[-200:]}")
            
            # Show examples being used
            # if examples:
            #     print(f"  📚 EXAMPLES BEING INJECTED:")
            #     for i, (role, content) in enumerate(examples[:2], 1):
            #         print(f"    {i}. {role}: {content[:100]}...")
            
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
    
    def _get_current_stage(self):
        """Get current stage from state machine"""
        if hasattr(self.state, 'state_machine') and self.state.state_machine:
            return self.state.state_machine.current_stage
        return 'unknown'
    
    def _get_simple_storyline_context(self, current_stage, current_state):
        """Get basic storyline context without over-engineering"""
        # Simple stage descriptions
        stage_contexts = {
            'onboarding': "🎯 PHASE 1: Du hilfst dem User dabei, Fake News am eigenen Leib zu erfahren und kritisches Denken zu entwickeln.",
            'stage_selection': "🎯 PHASE 2: Der User wählt seinen Vertiefungsbereich - Politik&Tech oder Psychologie&Gesellschaft.",
            'content_politics_tech': "🎯 PHASE 3: Vertiefung Politik & Technologie - Deepfakes, Wahlen, KI-Tools, demokratische Auswirkungen.",
            'content_psychology_society': "🎯 PHASE 3: Vertiefung Psychologie & Gesellschaft - Bias, Social Media, Manipulation, Filterblase.",
            'offboarding': "🎯 PHASE 4: Abschluss der Lernreise - Reflexion, Zusammenfassung, Ermutigung als 'Fake News Fighter'."
        }
        
        base_context = stage_contexts.get(current_stage, f"🎯 Unbekannte Phase: {current_stage}")
        
        # Add simple progress context if available
        if hasattr(self.state, 'state_machine') and self.state.state_machine:
            completed_stages = getattr(self.state.state_machine, 'completed_stages', [])
            if completed_stages:
                completed_names = []
                for stage in completed_stages:
                    if stage == 'content_politics_tech':
                        completed_names.append('Politik&Tech')
                    elif stage == 'content_psychology_society':
                        completed_names.append('Psychologie&Gesellschaft')
                
                if completed_names:
                    base_context += f" ✅ Bereits abgeschlossen: {', '.join(completed_names)}"
        
        return base_context
    
    def format_user_profile_for_llm(self, user_profile):
        """Format user profile for LLM prompt (graceful fallback)"""
        try:
            if not user_profile:
                return ""
            
            profile_parts = []
            
            # Basic info
            if user_profile.get('age'):
                profile_parts.append(f"Alter: {user_profile['age']}")
            if user_profile.get('fake_news_literacy', {}).get('self_assessed_skill'):
                skill = user_profile['fake_news_literacy']['self_assessed_skill']
                if skill != 'unknown':
                    profile_parts.append(f"Fake News Kenntnisse: {skill}")
            if user_profile.get('emotional_state', {}).get('current_mood'):
                mood = user_profile['emotional_state']['current_mood']
                if mood != 'unknown':
                    profile_parts.append(f"Stimmung: {mood}")
            
            if profile_parts:
                return f"BENUTZERPROFIL: {' | '.join(profile_parts)}"
            else:
                return ""
                
        except Exception as e:
            print(f"Error formatting user profile: {e}")
            return ""

    # TODO: Could be re-implemented for quality assurance later
    
    async def proactive_instruct(self):
        proactive_prompt = self.state.prompts['proactive_prompt']

        # Handle user profile context with graceful fallback
        user_profile_context = ""
        if hasattr(self.state, 'user_profile') and self.state.user_profile:
            try:
                user_profile_context = self.format_user_profile_for_llm(self.state.user_profile)
            except Exception as e:
                print(f"⚠️ Could not format user profile: {e}")
                user_profile_context = ""

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
                print(f"🔄 PROMPT_ADAPTION: Injecting context for state '{next_action.payload.get('current_state')}'")
                
                # Store state machine prompts and examples in agent state
                if 'state_system_prompts' in next_action.payload:
                    prompts = next_action.payload['state_system_prompts']
                    self.state.current_state_prompts = prompts
                    print(f"  ✅ Injected {len(prompts)} state prompts")
                if 'state_examples' in next_action.payload:
                    examples = next_action.payload['state_examples']
                    self.state.current_state_examples = examples
                    print(f"  ✅ Injected {len(examples)} state examples")
                
                # Debug: Show all payload keys 
                print(f"  📦 Payload keys: {list(next_action.payload.keys())}")
                
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
            # Use state information from Decision Agent (not from state machine directly)
            if next_action.type == NextActionDecisionType.PROMPT_ADAPTION and next_action.payload:
                # Decision Agent has already determined the correct state after transitions
                current_state = next_action.payload.get('current_state')
                print(f"🎯 USING DECISION AGENT STATE: {current_state}")
                
                # Get stage context from state machine for dynamic prompts only
                stage_context = {}
                if hasattr(self.state, 'state_machine') and self.state.state_machine:
                    stage_context = self.state.state_machine.get_state_context_for_decision_agent(self.state.conversation_turn_counter)
                    
                    # Check for unavailable stage requests in stage_selection
                    if current_state == 'stage_selection' and stage_context.get('unavailable_stage_requested'):
                        print(f"🚫 UNAVAILABLE STAGE REQUESTED: {stage_context['unavailable_stage_requested']}")
                    elif current_state == 'stage_selection' and stage_context.get('dynamic_stage_prompt'):
                        print(f"📋 DYNAMIC STAGE PROMPT AVAILABLE")
            else:
                # Fallback: Get current state from state machine
                current_state = self.get_current_state_from_state_machine()
                print(f"🎯 FALLBACK STATE MACHINE STATE: {current_state}")
                stage_context = {}
            
            # Build dynamic prompt with state-specific content and examples
            dynamic_prompt = self.build_dynamic_prompt(current_state, stage_context)
            
            # Use agent-specific model from config for dynamic conversations
            from config import config
            conversational_model = config.get("llm", "conversational_agent_model", fallback=config.get("llm", "model_name"))
            llm = llm_factory.get_llm(conversational_model)
            chain = dynamic_prompt | llm

            chat_chain = RunnableWithMessageHistory(
                chain,
                self.get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history"
            )
            
            # Handle user profile context with graceful fallback for async loading
            user_profile_context = ""
            if hasattr(self.state, 'user_profile') and self.state.user_profile:
                try:
                    user_profile_context = self.format_user_profile_for_llm(self.state.user_profile)
                except Exception as e:
                    print(f"⚠️ Could not format user profile: {e}")
                    user_profile_context = ""
            else:
                user_profile_context = "Benutzerprofil wird geladen..."
            
            # print(f"🎯 BEHAVIORAL INSTRUCTION: {getattr(self.state, 'current_guiding_instruction_name', 'None')}")
            # print(f"💭 COMBINED GUIDANCE: {getattr(self.state, 'current_guiding_instruction', 'None')[:100] if hasattr(self.state, 'current_guiding_instruction') else 'None'}...")
            # print(f"DEBUG: Using state-specific prompt and examples for state machine state: {current_state}")
            
            # Debug: Show what exact instructions the LLM gets (use injected prompts)
            if hasattr(self.state, 'current_state_prompts') and self.state.current_state_prompts:
                state_instructions = self.state.current_state_prompts
                print(f"🎭 EXACT STATE INSTRUCTIONS FROM DECISION AGENT ({len(state_instructions)} items):")
            else:
                state_prompts = self.state.prompts.get('state_system_prompts', {})
                state_instructions = state_prompts.get(current_state, [])
                print(f"🎭 FALLBACK STATE INSTRUCTIONS FROM LEGACY ({len(state_instructions)} items):")
            
            for i, instruction in enumerate(state_instructions, 1):
                print(f"   {i}. {instruction}")
            
            llm_answer_text = ""
            start_time = time.time()
            async for chunk in chat_chain.astream({
                "input": self.state.instruction,
                "user_profile_context": user_profile_context
            }, config=self.model_config):
                llm_answer_text += chunk.content
            
            llm_time = time.time() - start_time
            print(f"⚡ CONVERSATIONAL AGENT: {llm_time:.2f}s")
                
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
            
            # Use agent-specific model from config for streaming
            from config import config
            conversational_model = config.get("llm", "conversational_agent_model", fallback=config.get("llm", "model_name"))
            llm = llm_factory.get_llm(conversational_model)
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