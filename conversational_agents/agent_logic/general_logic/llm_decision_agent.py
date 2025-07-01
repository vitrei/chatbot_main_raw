import json
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage
from langchain_core.messages.ai import AIMessageChunk

from data_models.data_models import AgentState, NextActionDecision, NextActionDecisionType
from conversational_agents.agent_logic.base_decision_agent import BaseDecisionAgent
from large_language_models.llm_factory import llm_factory
from conversational_agents.agent_logic.state_machine_wrapper import create_state_machine_from_prompts, load_state_machine_config

from prompts.prompt_loader import prompt_loader

class LLMDecisionAgent(BaseDecisionAgent):

    def __init__(self):
        super().__init__()
        
        # Streamlined decision agent prompt for transitions
        decision_agent_prompt = """
Du bist Decision Agent für Fake-News-Aufklärung.

CURRENT STATE: {current_state}
USER MESSAGE: {last_user_message}
USER PROFILE: {user_profile}
TURN COUNTER: {turn_counter}

VERFÜGBARE TRANSITIONS:
{available_transitions}

TRANSITION REGELN FÜR {current_state}:
{transition_logic}

WICHTIG: Du MUSST IMMER eine Transition aus den verfügbaren Optionen wählen!
Es gibt KEINE Option für "keine Transition" - die State Machine funktioniert nur mit Transitions.

Analysiere die User-Antwort und wähle die passendste Transition:
- Bei Interesse/Neugier: wähle engagement/interest Transition
- Bei Ablehnung/Widerstand: wähle repair/comfort Transition  
- Bei Skepsis: wähle skeptical Transition
- Bei Verwirrung: wähle repair Transition

JSON Response:
{{
    "trigger": "gewählter_trigger_name",
    "reason": "warum diese Transition passt",
    "guiding_instruction": "general_guidance"
}}
"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Du bist ein intelligenter Decision Agent für Fake-News-Aufklärungsgespräche."),
            ("human", decision_agent_prompt),
        ])

        llm = llm_factory.get_llm()
        self.chain = prompt | llm 

    def get_state_machine_context(self, agent_state):
        """Extract state machine context from agent state"""
        if not hasattr(agent_state, 'state_machine') or not agent_state.state_machine:
            print("❌ No state machine found in agent_state")
            return {
                'current_state': 'init_greeting',  # default initial state
                'available_transitions': [],
                'state_specific_instructions': 'No state machine available - using default state',
                'fake_news_available': False,
                'fake_news_stimulus_url': None
            }
        
        sm = agent_state.state_machine
        # Pass turn counter for stage-aware progression
        context = sm.get_state_context_for_decision_agent(agent_state.conversation_turn_counter)
        
        # Show stage progression info
        stage_progress = context.get('stage_progress', {})
        print(f"📊 STAGE PROGRESS: {stage_progress.get('progress_percentage', 0):.1f}% (Turn {agent_state.conversation_turn_counter}/{stage_progress.get('target_turns', 15)})")
        
        # Show available vs stage-appropriate transitions
        all_transitions = context['available_transitions']
        stage_appropriate = context.get('stage_appropriate_transitions', all_transitions)
        print(f"🎯 TRANSITIONS: {len(stage_appropriate)} stage-appropriate of {len(all_transitions)} total")
        
        if len(stage_appropriate) < len(all_transitions):
            blocked = [t['trigger'] for t in all_transitions if not t.get('stage_appropriate', True)]
            print(f"⚠️ BLOCKED BY STAGE PROGRESSION: {blocked}")
        
        # Check for fake news stimulus availability
        fake_news_url = context.get('fake_news_stimulus_url')
        fake_news_available = fake_news_url is not None
        
        if fake_news_available:
            print(f"🎬 FAKE NEWS STIMULUS READY: {fake_news_url}")
        else:
            print("📭 No fake news stimulus available")
        
        # Get state-specific instructions from state machine config
        if hasattr(agent_state, 'state_machine') and agent_state.state_machine:
            from conversational_agents.agent_logic.state_machine_wrapper import load_state_machine_config
            state_machine_config = load_state_machine_config()
            if state_machine_config:
                state_prompts = state_machine_config.get('state_system_prompts', {})
                current_state_instructions = state_prompts.get(context['current_state'], ['No specific instructions'])
            else:
                current_state_instructions = ['No state machine config available']
        else:
            current_state_instructions = ['No state machine available']
        
        return {
            'current_state': context['current_state'],
            'available_transitions': context['available_transitions'],
            'state_specific_instructions': '\n'.join(current_state_instructions),
            'fake_news_available': fake_news_available,
            'fake_news_stimulus_url': fake_news_url
        }

    def get_transition_options_text(self, available_transitions):
        """Format available transitions for prompt"""
        if not available_transitions:
            return "Keine Transitions verfügbar"
        
        options = []
        for transition in available_transitions:
            options.append(f"- {transition['trigger']}: {transition['description']} -> {transition['dest']}")
        
        return '\n'.join(options)

    def get_transition_decision_logic(self, agent_state):
        """Extract transition decision logic ONLY for current state"""
        try:
            if not hasattr(agent_state, 'state_machine') or not agent_state.state_machine:
                return "Keine spezifische Transition-Logik verfügbar"
            
            current_state = agent_state.state_machine.get_current_state()
            current_stage = agent_state.state_machine.current_stage
            stages = agent_state.state_machine.stages
            
            if current_stage not in stages:
                return "Keine Stage-Konfiguration gefunden"
            
            stage_config = stages[current_stage]
            decision_logic = stage_config.get('transition_decision_logic', {})
            
            if not decision_logic:
                return "Keine Transition-Entscheidungslogik definiert"
            
            # ONLY get decision logic for current state
            current_state_logic = decision_logic.get(current_state, {})
            
            if not current_state_logic:
                return f"Keine spezifische Transition-Logik für State '{current_state}'"
            
            logic_text = [f"TRANSITION-ENTSCHEIDUNGEN FÜR {current_state.upper()}:"]
            for trigger, description in current_state_logic.items():
                logic_text.append(f"  • {trigger}: {description}")
            
            return '\n'.join(logic_text)
            
        except Exception as e:
            print(f"Error getting transition decision logic: {e}")
            return "Fehler beim Laden der Transition-Logik - verwende Standard-Entscheidungen"

    def get_last_user_message(self, chat_history_dict):
        """Extract the last user message from chat history"""
        for session_id, history in chat_history_dict.items():
            if history.messages:
                last_message = history.messages[-1]
                if isinstance(last_message, HumanMessage):
                    return last_message.content
        return "No user message found"

    def get_user_profile_info(self, agent_state):
        """Get user profile info from agent_state"""
        try:
            if hasattr(agent_state, 'user_profile') and agent_state.user_profile:
                return self.format_user_profile_for_prompt(agent_state.user_profile)
            return "KEIN BENUTZERPROFIL VERFÜGBAR - verwende Standard-Entscheidungslogik."
        except Exception as e:
            print(f"DEBUG: Could not get user profile from agent_state: {e}")
            return "FEHLER beim Laden des Benutzerprofils - verwende Standard-Entscheidungslogik."

    def format_user_profile_for_prompt(self, user_profile):
        """Format user profile data for the prompt"""
        if not user_profile:
            return "Kein Profil - Standard-Logik."
            
        profile_data = []
        
        if user_profile.get('age'):
            profile_data.append(f"Alter:{user_profile['age']}")
        if user_profile.get('fake_news_skill'):
            profile_data.append(f"FakeNewsSkill:{user_profile['fake_news_skill']}")
        if user_profile.get('attention_span'):
            profile_data.append(f"Aufmerksamkeit:{user_profile['attention_span']}")
        if user_profile.get('current_mood'):
            profile_data.append(f"Stimmung:{user_profile['current_mood']}")
        if user_profile.get('interaction_style'):
            profile_data.append(f"Stil:{user_profile['interaction_style']}")
        
        # Recommendations based on profile
        recommendations = []
        age = user_profile.get('age')
        if age and age < 16:
            recommendations.append("young_user_guidance")
        
        fake_news_skill = user_profile.get('fake_news_skill')
        if fake_news_skill == 'master':
            recommendations.append("expert_challenge")
        elif fake_news_skill == 'low':
            recommendations.append("beginner_support")
        
        current_mood = user_profile.get('current_mood')
        if current_mood == 'mad':
            recommendations.append("gentle_approach")
        
        attention_span = user_profile.get('attention_span')
        if attention_span == 'short':
            recommendations.append("quick_response")
        
        output_parts = []
        if profile_data:
            output_parts.append(f"PROFIL: {' | '.join(profile_data)}")
        if recommendations:
            output_parts.append(f"ZUSÄTZLICHE ANWEISUNGEN: {','.join(recommendations)}")
        
        return " || ".join(output_parts) if output_parts else "Profil leer - Standard-Logik."

    def add_state_machine_to_agent_state(self, agent_state):
        """Add state machine to existing AgentState"""
        
        if not hasattr(agent_state, 'state_machine') or not agent_state.state_machine:
            state_machine = create_state_machine_from_prompts(agent_state.prompts)
            agent_state.state_machine = state_machine
            print(f"🎰 State Machine added to AgentState: {state_machine.get_current_state() if state_machine else 'Failed'}")
        
        return agent_state
    
    def load_agent_state_context(self, agent_state: AgentState):
        """Load comprehensive context from AgentState"""
        try:
            # Get current state from state machine
            current_state = 'init_greeting'  # default
            available_transitions = []
            stage_info = {}
            
            if hasattr(agent_state, 'state_machine') and agent_state.state_machine:
                current_state = agent_state.state_machine.get_current_state()
                context = agent_state.state_machine.get_state_context_for_decision_agent(agent_state.conversation_turn_counter)
                available_transitions = context.get('available_transitions', [])
                stage_info = context.get('stage_progress', {})
            
            # Extract user profile information
            user_profile_summary = self.format_user_profile_for_decision(agent_state.user_profile)
            
            # Get current user input (prefer instruction over chat history)
            if hasattr(agent_state, 'instruction') and agent_state.instruction:
                last_user_message = agent_state.instruction
            else:
                last_user_message = self.get_last_user_message(agent_state.chat_history)
            
            return {
                'current_state': current_state,
                'user_profile': user_profile_summary,
                'conversation_turn_counter': agent_state.conversation_turn_counter,
                'user_id': agent_state.user_id,
                'last_user_message': last_user_message,
                'available_transitions': available_transitions,
                'stage_info': stage_info,
                'fake_news_available': hasattr(agent_state.state_machine, 'fake_news_stimulus_url') if agent_state.state_machine else False,
                'fake_news_stimulus_url': getattr(agent_state.state_machine, 'fake_news_stimulus_url', None) if agent_state.state_machine else None
            }
        except Exception as e:
            print(f"❌ Error loading agent state context: {e}")
            return {
                'current_state': 'init_greeting',
                'user_profile': 'No profile available',
                'conversation_turn_counter': agent_state.conversation_turn_counter,
                'user_id': agent_state.user_id,
                'last_user_message': 'No message found',
                'available_transitions': [],
                'stage_info': {},
                'fake_news_available': False,
                'fake_news_stimulus_url': None
            }
    
    def format_user_profile_for_decision(self, user_profile):
        """Format user profile for decision making context"""
        if not user_profile:
            return "No user profile available"
        
        profile_summary = []
        
        # Demographics
        if user_profile.get('age'):
            profile_summary.append(f"Age: {user_profile['age']}")
        if user_profile.get('school_type'):
            profile_summary.append(f"School: {user_profile['school_type']}")
        
        # Fake news literacy
        fake_news_skill = user_profile.get('fake_news_skill') or user_profile.get('self_assessed_skill')
        if fake_news_skill:
            profile_summary.append(f"FakeNews-Skill: {fake_news_skill}")
        
        # Emotional state
        if user_profile.get('current_mood'):
            profile_summary.append(f"Mood: {user_profile['current_mood']}")
        if user_profile.get('attention_span'):
            profile_summary.append(f"Attention: {user_profile['attention_span']}")
        
        # Interaction style
        if user_profile.get('interaction_style'):
            profile_summary.append(f"Style: {user_profile['interaction_style']}")
        
        return " | ".join(profile_summary) if profile_summary else "Basic profile only"
    
    def get_current_state_transitions(self, agent_state: AgentState):
        """Get available transitions for current state"""
        try:
            if not hasattr(agent_state, 'state_machine') or not agent_state.state_machine:
                return []
            
            available_transitions = agent_state.state_machine.get_available_transitions(agent_state.conversation_turn_counter)
            current_state = agent_state.state_machine.get_current_state()
            
            # Filter transitions to only show those from current state
            current_state_transitions = []
            for t in available_transitions:
                source = t.get('source')
                if (isinstance(source, str) and source == current_state) or \
                   (isinstance(source, list) and current_state in source) or \
                   (source == '*'):
                    current_state_transitions.append(t)
            
            print(f"🔄 AVAILABLE TRANSITIONS FOR {current_state}: {[t['trigger'] for t in current_state_transitions]}")
            return current_state_transitions
            
        except Exception as e:
            print(f"❌ Error getting transitions: {e}")
            return []
    
    def get_transition_decision_logic(self, agent_state: AgentState, current_state: str):
        """Get transition decision logic for current state"""
        try:
            state_machine_config = load_state_machine_config()
            if not state_machine_config:
                return "No transition logic available"
            
            stages = state_machine_config.get('stages', {})
            current_stage = 'onboarding'  # default
            
            if hasattr(agent_state, 'state_machine') and agent_state.state_machine:
                current_stage = agent_state.state_machine.current_stage
            
            if current_stage not in stages:
                return "No stage configuration found"
            
            decision_logic = stages[current_stage].get('transition_decision_logic', {})
            current_state_logic = decision_logic.get(current_state, {})
            
            if not current_state_logic:
                return f"No transition logic for state '{current_state}'"
            
            logic_text = []
            for trigger, description in current_state_logic.items():
                logic_text.append(f"• {trigger}: {description}")
            
            return '\n'.join(logic_text)
            
        except Exception as e:
            print(f"❌ Error getting transition logic: {e}")
            return "Error loading transition logic"
    
    def format_transitions_for_prompt(self, available_transitions):
        """Format available transitions for LLM prompt"""
        if not available_transitions:
            return "No transitions available"
        
        transition_text = []
        for t in available_transitions:
            transition_text.append(f"• {t['trigger']}: {t.get('description', 'No description')} → {t['dest']}")
        
        return '\n'.join(transition_text)
    
    def extract_and_parse_json(self, response_content):
        """Extract and parse JSON from LLM response"""
        try:
            # Clean response
            content = re.sub(r'```json\s*', '', response_content)
            content = re.sub(r'```\s*$', '', content)
            
            # Find JSON pattern
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            
            print("❌ No JSON found in response")
            return {}
            
        except Exception as e:
            print(f"❌ JSON parsing error: {e}")
            return {}
    
    def check_guard_rail_enforcement(self, agent_state: AgentState, available_transitions):
        """Check if guard rails require forced transitions"""
        try:
            if not hasattr(agent_state, 'state_machine') or not agent_state.state_machine:
                return None
            
            current_state = agent_state.state_machine.get_current_state()
            turn_counter = agent_state.conversation_turn_counter
            stage_info = agent_state.state_machine.stages.get(agent_state.state_machine.current_stage, {})
            
            mandatory_sequence = stage_info.get('mandatory_sequence', [])
            min_turns = stage_info.get('min_turns', 10)
            max_turns = stage_info.get('max_turns', 12)
            guard_rails = stage_info.get('guard_rails', {})
            
            # Force closure after max turns
            if turn_counter >= max_turns and guard_rails.get('force_closure_after_max_turns', False):
                closure_transitions = [t for t in available_transitions if t['dest'] == 'onboarding_closure']
                if closure_transitions:
                    return closure_transitions[0]['trigger']
            
            # Force progression through mandatory sequence
            if guard_rails.get('enforce_sequence', False) and mandatory_sequence:
                current_index = self.get_sequence_index_decision(current_state, mandatory_sequence)
                if current_index >= 0 and current_index < len(mandatory_sequence) - 1:
                    next_mandatory = mandatory_sequence[current_index + 1]
                    next_transitions = [t for t in available_transitions if t['dest'] == next_mandatory]
                    if next_transitions and turn_counter >= (current_index + 1) * 2:  # Force progression every 2 turns
                        return next_transitions[0]['trigger']
            
            return None
            
        except Exception as e:
            print(f"❌ Error in guard rail enforcement: {e}")
            return None
    
    def get_sequence_index_decision(self, state, sequence):
        """Get index of state in mandatory sequence (-1 if not in sequence)"""
        try:
            return sequence.index(state)
        except ValueError:
            return -1

    def next_action(self, agent_state: AgentState):
        print(f"🔢 TURN COUNTER: {agent_state.conversation_turn_counter}")
        print(f"👤 USER ID: {agent_state.user_id}")
        
        # Ensure state machine is initialized
        agent_state = self.add_state_machine_to_agent_state(agent_state)
        
        # Load comprehensive AgentState information
        agent_context = self.load_agent_state_context(agent_state)
        current_state = agent_context['current_state']
        
        print(f"🗨️ LAST USER MESSAGE: '{agent_context['last_user_message']}'")
        print(f"📊 USER PROFILE: {agent_context['user_profile']}")
        print(f"📝 AGENT STATE INSTRUCTION: '{agent_state.instruction}'")
        print(f"💬 CHAT HISTORY KEYS: {list(agent_state.chat_history.keys()) if agent_state.chat_history else 'None'}")
        
        # Debug chat history content
        if agent_state.chat_history:
            for session_id, history in agent_state.chat_history.items():
                print(f"📜 SESSION {session_id}: {len(history.messages) if hasattr(history, 'messages') else 0} messages")
                if hasattr(history, 'messages') and history.messages:
                    for i, msg in enumerate(history.messages[-3:]):  # Show last 3 messages
                        print(f"   {i}: {type(msg).__name__} - {msg.content[:50] if hasattr(msg, 'content') else 'No content'}...")
        
        # Get available transitions for current state
        available_transitions = self.get_current_state_transitions(agent_state)
        transition_logic = self.get_transition_decision_logic(agent_state, current_state)
        
        # Check for guard rail enforcement
        forced_transition = self.check_guard_rail_enforcement(agent_state, available_transitions)
        if forced_transition:
            print(f"⚡ GUARD RAIL ENFORCEMENT: Forcing transition {forced_transition}")
            if hasattr(agent_state, 'state_machine') and agent_state.state_machine:
                success = agent_state.state_machine.execute_transition(forced_transition, "Guard rail enforcement")
                if success:
                    current_state = agent_state.state_machine.get_current_state()
                    print(f"✅ FORCED TRANSITION EXECUTED: {forced_transition}")
        
        # Prepare LLM prompt data
        prompt_data = {
            "current_state": current_state,
            "last_user_message": agent_context['last_user_message'],
            "user_profile": agent_context['user_profile'],
            "turn_counter": agent_context['conversation_turn_counter'],
            "available_transitions": self.format_transitions_for_prompt(available_transitions),
            "transition_logic": transition_logic
        }
        
        # LLM call to decide on transition
        print(f"🤖 MAKING LLM DECISION CALL for state: {current_state}")
        print(f"📝 PROMPT DATA: {prompt_data}")
        
        try:
            response = self.chain.invoke(prompt_data)
            print(f"🤖 LLM RAW RESPONSE: {response.content}")
            
            llm_decision = self.extract_and_parse_json(response.content)
            print(f"🧠 LLM DECISION PARSED: {llm_decision}")
            
            # Execute state transition (LLM MUST always choose one)
            trigger = llm_decision.get('trigger')
            reason = llm_decision.get('reason', 'LLM decision')
            
            if not trigger:
                print(f"❌ LLM FAILED TO CHOOSE TRANSITION - using fallback")
                # Fallback: choose first available transition
                if available_transitions:
                    trigger = available_transitions[0]['trigger']
                    reason = "Fallback - LLM failed to choose"
                    print(f"🔄 FALLBACK TRANSITION: {trigger}")
            
            if trigger:
                print(f"🔄 LLM RECOMMENDS TRANSITION: {trigger} - {reason}")
                
                if hasattr(agent_state, 'state_machine') and agent_state.state_machine:
                    success = agent_state.state_machine.execute_transition(trigger, reason)
                    if success:
                        print(f"✅ STATE TRANSITION EXECUTED: {trigger} - {reason}")
                        # Update current state after transition
                        current_state = agent_state.state_machine.get_current_state()
                    else:
                        print(f"❌ STATE TRANSITION FAILED: {trigger}")
            else:
                print(f"🚫 NO TRANSITION POSSIBLE for {current_state}")
            
        except Exception as e:
            print(f"❌ LLM Decision failed: {e}")
            # Fallback: force a transition
            if available_transitions:
                trigger = available_transitions[0]['trigger']
                reason = "Exception fallback"
                print(f"🔄 EXCEPTION FALLBACK TRANSITION: {trigger}")
                if hasattr(agent_state, 'state_machine') and agent_state.state_machine:
                    agent_state.state_machine.execute_transition(trigger, reason)
                    current_state = agent_state.state_machine.get_current_state()
            llm_decision = {'guiding_instruction': 'general_guidance'}
        
        # Load state-specific content from state_machine.json
        state_machine_config = load_state_machine_config()
        current_state_prompts = []
        current_state_examples = []
        
        if state_machine_config:
            state_prompts = state_machine_config.get('state_system_prompts', {})
            state_examples = state_machine_config.get('state_examples', {})
            current_state_prompts = state_prompts.get(current_state, [])
            current_state_examples = state_examples.get(current_state, [])
            
            print(f"🎭 LOADED STATE PROMPTS: {len(current_state_prompts)} for {current_state}")
            print(f"📚 LOADED STATE EXAMPLES: {len(current_state_examples)} for {current_state}")
        
        # Return PROMPT_ADAPTION with all context
        return NextActionDecision(
            type=NextActionDecisionType.PROMPT_ADAPTION,
            action="inject_complete_context",
            payload={
                # State Machine Context
                'current_state': current_state,
                'state_system_prompts': current_state_prompts,
                'state_examples': current_state_examples,
                
                # AgentState Context
                'user_profile': agent_context['user_profile'],
                'conversation_turn_counter': agent_context['conversation_turn_counter'],
                'user_id': agent_context['user_id'],
                'last_user_message': agent_context['last_user_message'],
                
                # Additional Context
                'available_transitions': available_transitions,
                'stage_info': agent_context['stage_info'],
                'fake_news_available': agent_context.get('fake_news_available', False),
                'fake_news_url': agent_context.get('fake_news_stimulus_url'),
                
                # Guiding instruction from LLM decision
                'guiding_instruction': llm_decision.get('guiding_instruction', 'general_guidance')
            }
        )

    

