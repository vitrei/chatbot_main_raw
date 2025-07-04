"""
Helper functions for the LLM Decision Agent.
These functions handle context extraction, formatting, and data processing.
"""

import json
import re
import time
from langchain.schema import HumanMessage, AIMessage
from conversational_agents.agent_logic.state_machine_wrapper import load_state_machine_config


class DecisionAgentHelpers:
    """Utility class containing helper methods for decision agent operations."""
    
    @staticmethod
    def get_last_user_message(chat_history_dict):
        """Extract the last user message from chat history"""
        for session_id, history in chat_history_dict.items():
            if history.messages:
                last_message = history.messages[-1]
                if isinstance(last_message, HumanMessage):
                    return last_message.content
        return "No user message found"
    
    @staticmethod
    def format_user_profile_for_prompt(user_profile):
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

    @staticmethod
    def format_user_profile_for_decision(user_profile):
        """Format user profile for decision making context"""
        if not user_profile:
            return "Profil wird geladen (Standard-Entscheidungen verwenden)"
        
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

    @staticmethod
    def get_current_state_transitions(agent_state):
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
            
            return current_state_transitions
            
        except Exception as e:
            print(f"❌ Error getting transitions: {e}")
            return []

    @staticmethod
    def get_transition_decision_logic(agent_state, current_state):
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

    @staticmethod
    def format_transitions_for_prompt(available_transitions):
        """Format available transitions for LLM prompt"""
        if not available_transitions:
            return "No transitions available"
        
        transition_text = []
        for t in available_transitions:
            transition_text.append(f"• {t['trigger']}: {t.get('description', 'No description')} → {t['dest']}")
        
        return '\n'.join(transition_text)

    @staticmethod
    def get_stage_turn_info(agent_state) -> str:
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

    @staticmethod
    def get_stage_context_short(agent_state) -> str:
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

    @staticmethod
    def get_state_purpose_short(agent_state, current_state: str) -> str:
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

    @staticmethod
    def extract_trigger_fast(response_content):
        """ULTRA-FAST trigger extraction - minimal parsing"""
        try:
            # Quick regex for trigger only
            trigger_match = re.search(r'"trigger":\s*"([^"]+)"', response_content)
            if trigger_match:
                return {"trigger": trigger_match.group(1)}
            
            # Fallback to full parsing if needed
            return DecisionAgentHelpers.extract_and_parse_json(response_content)
        except:
            return {"trigger": "emergency_closure"}

    @staticmethod
    def extract_and_parse_json(response_content):
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

    @staticmethod
    def load_agent_state_context(agent_state):
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
            user_profile_summary = DecisionAgentHelpers.format_user_profile_for_decision(agent_state.user_profile)
            
            # Get current user input (prefer instruction over chat history)
            if hasattr(agent_state, 'instruction') and agent_state.instruction:
                last_user_message = agent_state.instruction
            else:
                last_user_message = DecisionAgentHelpers.get_last_user_message(agent_state.chat_history)
            
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