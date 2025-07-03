import json
import re
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage
from langchain_core.messages.ai import AIMessageChunk

from data_models.data_models import AgentState, NextActionDecision, NextActionDecisionType
from conversational_agents.agent_logic.base_decision_agent import BaseDecisionAgent
from large_language_models.llm_factory import llm_factory
from conversational_agents.agent_logic.state_machine_wrapper import create_state_machine_from_prompts, load_state_machine_config
from conversational_agents.agent_logic.general_logic.decision_rule_engine import DecisionRuleEngine

from prompts.prompt_loader import prompt_loader

class LLMDecisionAgent(BaseDecisionAgent):

    def __init__(self):
        # Initialize rule engine (will be set up when config is loaded)
        self.rule_engine = None
        super().__init__()
        
        # ULTRA-COMPACT decision agent prompt for speed  
        decision_agent_prompt = """
Du bist Decision Agent. SCHNELL antworten!

=== KONTEXT ===
STATE: {current_state} | STAGE: {current_stage} | TURN: {turn_counter} ({stage_turn_info})
USER: {last_user_message}
PROFIL: {user_profile}

=== ZIELE ===
Stage: {stage_context_short}
State: {state_purpose_short}

=== TRANSITIONS ===
{available_transitions}

=== REGELN ===
{transition_logic}

=== ENTSCHEIDUNGSFRAMEWORK ===
1. User-Absicht verstehen:
   - "erzähl mir" = Vertiefung (meist kein Transition)
   - Themenwechsel = Transition wählen
   - Verwirrung = repair/comfort

2. Pädagogisches Ziel:
   - Vertiefung vs. Progression
   - User Engagement beachten

WICHTIG: Antworte NUR mit diesem JSON-Format:
{{
    "trigger": "exact_trigger_name",
    "reason": "Detaillierte Begründung basierend auf User-Absicht",
    "user_intent": "Was der User wirklich möchte",
    "pedagogical_goal": "Warum diese Transition das Lernziel unterstützt"
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
            # print(f"🎰 State Machine added to AgentState: {state_machine.get_current_state() if state_machine else 'Failed'}")
        
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
            
            # Available transitions computed
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
    
    def extract_trigger_fast(self, response_content):
        """ULTRA-FAST trigger extraction - minimal parsing"""
        try:
            # Quick regex for trigger only
            trigger_match = re.search(r'"trigger":\s*"([^"]+)"', response_content)
            if trigger_match:
                return {"trigger": trigger_match.group(1)}
            
            # Fallback to full parsing if needed
            return self.extract_and_parse_json(response_content)
        except:
            return {"trigger": "emergency_closure"}
    
    def extract_and_parse_json(self, response_content):
        """Extract and parse JSON from LLM response - robust version"""
        try:
            # Clean response - remove markdown formatting
            content = re.sub(r'```json\s*', '', response_content)
            content = re.sub(r'```\s*$', '', content)
            content = content.strip()
            
            # Look for JSON in different patterns
            json_patterns = [
                r'\{[^{}]*"trigger"[^{}]*\}',  # Look for trigger field specifically
                r'\{.*?"trigger".*?\}',        # More flexible trigger search
                r'\{.*\}',                     # Any JSON-like structure
            ]
            
            for pattern in json_patterns:
                json_match = re.search(pattern, content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    try:
                        parsed = json.loads(json_str)
                        if 'trigger' in parsed:  # Validate it has required field
                            return parsed
                    except json.JSONDecodeError:
                        continue
            
            # Fallback: Try to extract trigger manually from text
            print("🔄 JSON parsing failed, trying manual extraction...")
            trigger_match = re.search(r'"trigger":\s*"([^"]+)"', content)
            if trigger_match:
                trigger = trigger_match.group(1)
                print(f"🔧 Extracted trigger manually: {trigger}")
                return {
                    'trigger': trigger,
                    'reason': 'Manual extraction from malformed response',
                    'user_intent': 'Parsing fallback',
                    'pedagogical_goal': 'Continue conversation'
                }
            
            print("❌ No JSON or trigger found in response")
            print(f"❌ Response content: {response_content[:200]}...")
            return {}
            
        except Exception as e:
            print(f"❌ JSON parsing error: {e}")
            print(f"❌ Response content: {response_content[:200]}...")
            return {}
    
    def check_guard_rail_enforcement(self, agent_state: AgentState, available_transitions):
        """Check if guard rails require forced transitions using rule engine"""
        try:
            if not hasattr(agent_state, 'state_machine') or not agent_state.state_machine:
                print("❌ No state machine available for guard rail check")
                return None
            
            # Initialize rule engine if not already done
            if self.rule_engine is None:
                config = load_state_machine_config()
                if config:
                    self.rule_engine = DecisionRuleEngine(config)
                    print(f"🎯 Rule engine initialized with {len(self.rule_engine.rules)} rules")
                else:
                    print("❌ Could not load config for rule engine")
                    return None
            
            # Get context for rule evaluation
            context = self.rule_engine.get_context_from_agent_state(agent_state, available_transitions)
            
            if 'error' in context:
                print(f"❌ Context error: {context['error']}")
                return None
            
            # Evaluate rules
            decision = self.rule_engine.evaluate_decision(context)
            
            if decision['forced']:
                print(f"🎯 RULE ENGINE DECISION: {decision['trigger']} (reason: {decision['reason']})")
                return decision['trigger']
            else:
                print(f"✅ No forced transitions needed - LLM can decide")
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
    
    def get_stage_turn_info(self, agent_state: AgentState) -> str:
        """Get stage-relative turn information"""
        try:
            if not hasattr(agent_state, 'state_machine') or not agent_state.state_machine:
                return "unknown"
            
            sm = agent_state.state_machine
            turn_counter = agent_state.conversation_turn_counter
            stage_start_turn = getattr(sm, 'stage_start_turn', 0)
            turns_in_stage = turn_counter - stage_start_turn
            
            stage_config = sm.stages.get(sm.current_stage, {})
            max_turns_in_stage = stage_config.get('max_turns_in_stage', stage_config.get('max_turns', 15))
            
            return f"{turns_in_stage}/{max_turns_in_stage}"
        except Exception as e:
            return f"error: {e}"
    
    def get_stage_context_description(self, agent_state: AgentState) -> str:
        """Get description of current stage goals and context"""
        try:
            if not hasattr(agent_state, 'state_machine') or not agent_state.state_machine:
                return "Keine Stage-Information verfügbar"
            
            current_stage = agent_state.state_machine.current_stage
            
            stage_descriptions = {
                'onboarding': """
**ONBOARDING STAGE ZIEL**: User für Fake News sensibilisieren durch persönliches Erlebnis
- Fake Video zeigen → Reaktion abwarten → kritisches Denken fördern
- Emotionale Betroffenheit nutzen für Lernmoment
- Übergang zu Content-Stages vorbereiten""",
                
                'stage_selection': """
**STAGE SELECTION ZIEL**: User zwischen zwei Content-Bereichen wählen lassen
- Politik & Technologie vs. Psychologie & Gesellschaft
- User-Interesse identifizieren für personalisiertes Lernen""",
                
                'content_politics_tech': """
**POLITIK & TECHNOLOGIE STAGE ZIEL**: Verständnis für technische und politische Aspekte
- Technologie: Deepfakes, KI-Tools, Manipulation-Software
- Politik: Wahlbeeinflussung, Propaganda, demokratische Auswirkungen
- Synthese: Verbindung zwischen Technik und politischer Nutzung""",
                
                'content_psychology_society': """
**PSYCHOLOGIE & GESELLSCHAFT STAGE ZIEL**: Verständnis für menschliche und gesellschaftliche Faktoren
- Psychologie: Cognitive Bias, Bestätigungsfehler, emotionale Manipulation
- Gesellschaft: Social Media Dynamiken, Filterblase, Polarisierung
- Synthese: Wie psychologische Faktoren gesellschaftliche Probleme verstärken"""
            }
            
            return stage_descriptions.get(current_stage, f"Unbekannte Stage: {current_stage}")
            
        except Exception as e:
            return f"Fehler beim Laden der Stage-Beschreibung: {e}"
    
    def get_stage_context_short(self, agent_state: AgentState) -> str:
        """Get compact stage context for fast decisions"""
        try:
            if not hasattr(agent_state, 'state_machine') or not agent_state.state_machine:
                return "Unbekannt"
            
            current_stage = agent_state.state_machine.current_stage
            
            short_descriptions = {
                'onboarding': "Fake News Sensibilisierung durch persönliches Video",
                'stage_selection': "Wahl zwischen Politik/Tech vs Psychologie/Gesellschaft", 
                'content_politics_tech': "Technologie + Politik: Deepfakes, Wahlen, Propaganda",
                'content_psychology_society': "Psychologie + Gesellschaft: Bias, Social Media, Filterblase"
            }
            
            return short_descriptions.get(current_stage, current_stage)
            
        except Exception as e:
            return "Error"
    
    def get_state_purpose_description(self, agent_state: AgentState, current_state: str) -> str:
        """Get description of what the current state is trying to achieve"""
        try:
            if not hasattr(agent_state, 'state_machine') or not agent_state.state_machine:
                return "Keine State-Information verfügbar"
            
            # Load state machine config for state descriptions
            from conversational_agents.agent_logic.state_machine_wrapper import load_state_machine_config
            config = load_state_machine_config()
            
            if not config:
                return "State Machine Konfiguration nicht verfügbar"
            
            # Get state examples/descriptions which describe the purpose
            state_examples = config.get('state_examples', {})
            current_state_examples = state_examples.get(current_state, [])
            
            if current_state_examples:
                # Use first example as purpose description
                first_example = current_state_examples[0] if isinstance(current_state_examples, list) else current_state_examples
                if isinstance(first_example, dict) and 'bot' in first_example:
                    return f"**{current_state.upper()} ZWECK**: {first_example['bot'][:100]}..."
                elif isinstance(first_example, str):
                    return f"**{current_state.upper()} ZWECK**: {first_example[:100]}..."
            
            # Fallback: use state system prompts
            state_prompts = config.get('state_system_prompts', {})
            current_state_prompts = state_prompts.get(current_state, [])
            
            if current_state_prompts and isinstance(current_state_prompts, list) and current_state_prompts:
                return f"**{current_state.upper()} ZWECK**: {' '.join(current_state_prompts[:2])}"
            
            return f"**{current_state.upper()}**: Zweck nicht definiert in Konfiguration"
            
        except Exception as e:
            return f"Fehler beim Laden der State-Beschreibung: {e}"
    
    def get_state_purpose_short(self, agent_state: AgentState, current_state: str) -> str:
        """Get compact state purpose for fast decisions"""
        try:
            # Quick hardcoded mappings for speed
            state_purposes = {
                'content_intro_pt': "Politik/Tech Einstieg - Thema wählen",
                'politics_deep_dive': "Politik vertiefen - Wahlen, Propaganda",
                'technology_deep_dive': "Technologie vertiefen - Deepfakes, KI",
                'content_synthesis_pt': "Politik+Tech zusammenfassen",
                'content_intro_ps': "Psychologie/Gesellschaft Einstieg",
                'psychology_deep_dive': "Psychologie vertiefen - Bias, Manipulation", 
                'society_deep_dive': "Gesellschaft vertiefen - Social Media",
                'content_synthesis_ps': "Psychologie+Gesellschaft zusammenfassen",
                'stage_selection': "Content-Bereich wählen",
                'onboarding_closure': "Onboarding abschließen"
            }
            
            return state_purposes.get(current_state, current_state)
            
        except Exception as e:
            return current_state

    def next_action(self, agent_state: AgentState):
        print(f"Turn: {agent_state.conversation_turn_counter}, User: {agent_state.user_id}")
        
        # Ensure state machine is initialized
        agent_state = self.add_state_machine_to_agent_state(agent_state)
        
        # Load comprehensive AgentState information
        agent_context = self.load_agent_state_context(agent_state)
        current_state = agent_context['current_state']
        
        # Get available transitions for current state
        available_transitions = self.get_current_state_transitions(agent_state)
        transition_logic = self.get_transition_decision_logic(agent_state, current_state)
        
        # Check for guard rail enforcement
        forced_transition = self.check_guard_rail_enforcement(agent_state, available_transitions)
        if forced_transition:
            print(f"⚡ GUARD RAIL ENFORCEMENT: Forcing transition {forced_transition}")
            if hasattr(agent_state, 'state_machine') and agent_state.state_machine:
                success = agent_state.state_machine.execute_transition(forced_transition, "Guard rail enforcement", agent_state.conversation_turn_counter)
                if success:
                    current_state = agent_state.state_machine.get_current_state()
                    print(f"✅ FORCED TRANSITION EXECUTED: {forced_transition}")
                    # CRITICAL FIX: Reload transitions after forced transition
                    available_transitions = self.get_current_state_transitions(agent_state)
                    transition_logic = self.get_transition_decision_logic(agent_state, current_state)
                    print(f"🔄 RELOADED TRANSITIONS FOR NEW STATE: {current_state}")
        
        # Prepare FAST LLM prompt data - compact versions for speed
        current_stage = agent_state.state_machine.current_stage if hasattr(agent_state, 'state_machine') and agent_state.state_machine else 'unknown'
        stage_turn_info = self.get_stage_turn_info(agent_state)
        stage_context_short = self.get_stage_context_short(agent_state)
        state_purpose_short = self.get_state_purpose_short(agent_state, current_state)
        
        prompt_data = {
            "current_state": current_state,
            "current_stage": current_stage,
            "stage_turn_info": stage_turn_info,
            "last_user_message": agent_context['last_user_message'],
            "user_profile": agent_context['user_profile'],
            "turn_counter": agent_context['conversation_turn_counter'],
            "stage_context_short": stage_context_short,
            "state_purpose_short": state_purpose_short,
            "available_transitions": self.format_transitions_for_prompt(available_transitions),
            "transition_logic": transition_logic
        }
        
        # LLM call to decide on transition - FAST MODE
        print("=== LLM CALL ===")
        print(f"{current_state}")
        
        try:
            start_time = time.time()
            response = self.chain.invoke(prompt_data)
            llm_time = time.time() - start_time
            
            # FAST JSON parsing - just get trigger
            llm_decision = self.extract_trigger_fast(response.content)
            print(f"⚡ LLM: {llm_time:.2f}s")
            
            # Execute state transition (LLM MUST always choose one)
            trigger = llm_decision.get('trigger')
            
            if not trigger:
                # Fallback: choose first allowed transition
                allowed_transitions_fallback = [t for t in available_transitions if t.get('allowed', True)]
                if allowed_transitions_fallback:
                    trigger = allowed_transitions_fallback[0]['trigger']
                    print(f"🔄 FALLBACK → {trigger}")
            
            if trigger:
                print(f"🔄 LLM → {trigger}")
                
                if hasattr(agent_state, 'state_machine') and agent_state.state_machine:
                    # Quick validation
                    allowed_transitions = [t for t in available_transitions if t.get('allowed', True)]
                    chosen_transition = next((t for t in allowed_transitions if t['trigger'] == trigger), None)
                    
                    if chosen_transition:
                        # Execute allowed transition
                        success = agent_state.state_machine.execute_transition(trigger, "LLM decision", agent_state.conversation_turn_counter)
                        if success:
                            current_state = agent_state.state_machine.get_current_state()
                    else:
                        # Handle blocked transitions
                        current_stage = agent_state.state_machine.current_stage
                        
                        # Special handling for stage_selection
                        if current_stage == 'stage_selection':
                            return NextActionDecision(
                                type=NextActionDecisionType.GENERATE_ANSWER, 
                                action="explain_unavailable_stage",
                                payload={"blocked_stage": trigger}
                            )
                        
                        # Fallback for other stages
                        if allowed_transitions:
                            fallback_trigger = allowed_transitions[0]['trigger']
                            agent_state.state_machine.execute_transition(fallback_trigger, "Fallback", agent_state.conversation_turn_counter)
                            current_state = agent_state.state_machine.get_current_state()
            else:
                print(f"🚫 NO TRANSITION POSSIBLE for {current_state}")
            
        except Exception as e:
            print(f"❌ LLM Decision failed: {e}")
            # Exception fallback: use first allowed transition
            allowed_transitions_exception = [t for t in available_transitions if t.get('allowed', True)]
            if allowed_transitions_exception:
                trigger = allowed_transitions_exception[0]['trigger']
                reason = "Exception fallback - LLM failed"
                print(f"🔄 EXCEPTION FALLBACK TRANSITION: {trigger}")
                if hasattr(agent_state, 'state_machine') and agent_state.state_machine:
                    success = agent_state.state_machine.execute_transition(trigger, reason, agent_state.conversation_turn_counter)
                    if success:
                        current_state = agent_state.state_machine.get_current_state()
                        print(f"✅ EXCEPTION FALLBACK EXECUTED: {trigger}")
                    else:
                        print(f"❌ EXCEPTION FALLBACK FAILED: {trigger}")
            else:
                print(f"❌ NO ALLOWED TRANSITIONS FOR EXCEPTION FALLBACK")
                print(f"    Available transitions: {[t['trigger'] for t in available_transitions]}")
                print(f"    All blocked by guard rails")
            llm_decision = {'guiding_instruction': 'general_guidance'}
            trigger = None  # Ensure trigger is set for safety
        
        # Load state-specific content from state_machine.json
        state_machine_config = load_state_machine_config()
        current_state_prompts = []
        current_state_examples = []
        
        if state_machine_config:
            state_prompts = state_machine_config.get('state_system_prompts', {})
            state_examples = state_machine_config.get('state_examples', {})
            current_state_prompts = state_prompts.get(current_state, [])
            current_state_examples = state_examples.get(current_state, [])
            
            # print(f"🎭 LOADED STATE PROMPTS: {len(current_state_prompts)} for {current_state}")
            # print(f"📚 LOADED STATE EXAMPLES: {len(current_state_examples)} for {current_state}")
        
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

    

