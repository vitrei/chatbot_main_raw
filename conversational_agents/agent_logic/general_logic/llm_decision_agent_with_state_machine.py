import json
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage
from langchain_core.messages.ai import AIMessageChunk

from data_models.data_models import AgentState, NextActionDecision, NextActionDecisionType
from conversational_agents.agent_logic.base_decision_agent import BaseDecisionAgent
from large_language_models.llm_factory import llm_factory
from prompts.prompt_loader import prompt_loader

from dependency_injection import StateMachineManager

class LLMDecisionAgent(BaseDecisionAgent):
    def __init__(self, state_machine_manager: StateMachineManager):
        super().__init__()
        self.state_machine_manager = state_machine_manager
        
        decision_agent_prompt = """
Der Chatbot ist definiert durch folgenden Prompt:
{system_prompt}

Das ist der Dialog zwischen dem Chatbot und einem Menschen:
{chat_history}

{user_profile_info}

AKTUELLER STATE MACHINE KONTEXT:
{state_machine_context}

WICHTIG: Berücksichtige das Benutzerprofil UND den aktuellen State bei der Entscheidung!

Der Chatbot soll nun die nächste sinnvolle Aktion ausführen. Mögliche Aktionen sind:
    GENERATE_ANSWER: Direkt eine Antwort generieren.
    GUIDING_INSTRUCTIONS: Den Dialog in eine bestimmte Richtung lenken.
    STATE_TRANSITION: Zu einem anderen State wechseln.
    ACTION: Eine externe Funktion aufrufen.

Mögliche GUIDING_INSTRUCTIONS mit key und description sind:
    {guiding_instructions}

MÖGLICHE STATE TRANSITIONS:
{possible_transitions}

Mögliche ACTION mit key und description sind:
    {actions}

ENTSCHEIDUNGSHILFEN basierend auf State Machine:
- Prüfe ob der aktuelle Dialog-Verlauf einen State-Wechsel erfordert
- Berücksichtige die State-Description für passende Reaktionen
- Bei User-Signalen für Beendigung: Transition zu entsprechendem State

Du gibst deine Antwort als JSON in folgender Weise:

{{
    "next_action": "GENERATE_ANSWER"
}}

oder

{{
    "next_action": "GUIDING_INSTRUCTIONS",
    "type": "<key>"
}}

oder 

{{
    "next_action": "STATE_TRANSITION",
    "type": "<target_state>"
}}

oder

{{
    "next_action": "ACTION",
    "type": "<key>"
}}
"""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Du bist ein intelligenter Decision Agent und wählst für eine Konversation zwischen einem Chatbot und einem Menschen die beste Aktion basierend auf dem Benutzerprofil und Gesprächskontext."),
                ("human", decision_agent_prompt),
            ]
        )

        llm = llm_factory.get_llm()
        self.chain = prompt | llm 

    def get_user_profile_info(self, agent_state):
        """Get user profile info from agent_state (populated by pre-processor)"""
        try:
            if hasattr(agent_state, 'user_profile') and agent_state.user_profile:
                return self.format_user_profile_for_prompt(agent_state.user_profile)
            return "KEIN BENUTZERPROFIL VERFÜGBAR - verwende Standard-Entscheidungslogik."
        except Exception as e:
            print(f"DEBUG: Could not get user profile from agent_state: {e}")
            return "FEHLER beim Laden des Benutzerprofils - verwende Standard-Entscheidungslogik."

    def format_user_profile_for_prompt(self, user_profile):
        """Format user profile data for the prompt - GLEICHE LOGIK, komprimiertes Output"""
        if not user_profile:
            return "Kein Profil - Standard-Logik."
            
        profile_data = []
        
        # Add available profile information (gleiche Logik)
        if user_profile.get('age'):
            profile_data.append(f"Alter:{user_profile['age']}")
        if user_profile.get('gender'):
            profile_data.append(f"Geschlecht:{user_profile['gender']}")
        if user_profile.get('school_type'):
            profile_data.append(f"Schule:{user_profile['school_type']}")
        if user_profile.get('region'):
            profile_data.append(f"Region:{user_profile['region']}")
        if user_profile.get('social_media_usage'):
            profile_data.append(f"SocialMedia:{user_profile['social_media_usage']}")
        if user_profile.get('fake_news_skill'):
            profile_data.append(f"FakeNewsSkill:{user_profile['fake_news_skill']}")
        if user_profile.get('fact_checking_habits'):
            profile_data.append(f"Factcheck:{user_profile['fact_checking_habits']}")
        if user_profile.get('vocabulary_level'):
            profile_data.append(f"Vokabular:{user_profile['vocabulary_level']}")
        if user_profile.get('interaction_style'):
            profile_data.append(f"Stil:{user_profile['interaction_style']}")
        if user_profile.get('attention_span'):
            profile_data.append(f"Aufmerksamkeit:{user_profile['attention_span']}")
        if user_profile.get('current_mood'):
            profile_data.append(f"Stimmung:{user_profile['current_mood']}")
        if user_profile.get('interests'):
            interests_str = ",".join(user_profile['interests'][:3])  # Nur erste 3 Interessen
            profile_data.append(f"Interessen:{interests_str}")
        
        recommendations = []
        
        age = user_profile.get('age')
        if age:
            if age < 16:
                recommendations.append("young_user_guidance")
            elif age < 18:
                recommendations.append("lockere_sprache")
        
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
        
        if profile_data or recommendations:
            output_parts = []
            
            if profile_data:
                output_parts.append(f"PROFIL: {' | '.join(profile_data)}")
            
            if recommendations:
                output_parts.append(f"AKTIONEN: {','.join(recommendations)}")
            
            turn_hint = "Turn0-1:source_check/skepticism, Turn2+:emotional_content"
            output_parts.append(f"STRATEGIE: {turn_hint}")
            
            return " || ".join(output_parts)
        else:
            return "Profil leer - Standard-Logik."

    def get_state_machine_context(self) -> str:
        """Format current state machine context for prompt"""
        current_info = self.state_machine_manager.get_current_state_info()
        
        if not current_info:
            return "Kein aktiver State Machine Kontext"
            
        context_parts = [
            f"Aktueller State: {current_info['state_id']} ({current_info['name']})",
            f"State Beschreibung: {current_info['description']}"
        ]
        
        transitions = current_info.get('transitions', [])
        if transitions:
            context_parts.append(f"Mögliche Übergänge: {', '.join(transitions)}")
            
        return " | ".join(context_parts)
    
    def get_possible_transitions_text(self) -> str:
        """Get formatted text of possible transitions with descriptions"""
        possible_states = self.state_machine_manager.get_possible_transitions()
        
        if not possible_states:
            return "Keine State-Transitions verfügbar"
            
        machine = self.state_machine_manager.state_machines[self.state_machine_manager.current_machine]
        
        transition_info = []
        for state_id in possible_states:
            state_info = machine['states'].get(state_id, {})
            name = state_info.get('name', state_id)
            desc = state_info.get('description', '')[:100] + "..." if len(state_info.get('description', '')) > 100 else state_info.get('description', '')
            transition_info.append(f"{state_id} ({name}): {desc}")
            
        return "\n".join(transition_info)

    def next_action(self, agent_state: AgentState):    
        user_profile_info = self.get_user_profile_info(agent_state)
        state_machine_context = self.get_state_machine_context()
        possible_transitions = self.get_possible_transitions_text()

        current_info = self.state_machine_manager.get_current_state_info()
        print(f"\n🤖 DECISION AGENT - Turn {agent_state.conversation_turn_counter}")
        print(f"📍 Current State: {current_info['state_id']} ({current_info['name']})")
        print(f"📝 State Behavior: {current_info['description'][:100]}...")
        print(f"🎯 Available Transitions: {current_info['transitions']}")
        
        prompts = prompt_loader.get_all_prompts()
        system_prompt = prompts['system_prompt']
        guiding_instruction_prompts = prompts['guiding_instructions']
        guidings_instructions_str = "" 
        for key, value in guiding_instruction_prompts.items():
            guidings_instructions_str += f"{key}: {value}\n"

        actions = """Keine spezifischen Actions definiert für Fake News Gespräche."""
        chat_history = self.generate_dialog(agent_state.chat_history, agent_state.instruction)
        
        # print("🔍 User profile info for LLM:", user_profile_info if user_profile_info else "None available")
        # print("🔍 Chat history:", chat_history)
        # print(f"🔍 Turn counter: {agent_state.conversation_turn_counter}")

        response = self.chain.invoke({
            "system_prompt": system_prompt,
            "chat_history": chat_history,
            "user_profile_info": user_profile_info,
            "state_machine_context": state_machine_context,
            "possible_transitions": possible_transitions,
            "guiding_instructions": guidings_instructions_str,
            "actions": actions
        })

        response_json = response.content

        while response_json == None or not self.is_json_parsable(response_json):
            print("Not a valid JSON. Retrying...")
            response = self.chain.invoke(
                {
                    "system_prompt": system_prompt,
                    "chat_history": chat_history,
                    "user_profile_info": user_profile_info,
                    "state_machine_context": state_machine_context,
                    "possible_transitions": possible_transitions,
                    "guiding_instructions": guidings_instructions_str,
                    "actions": actions
                }
            )
            response_json = self.extract_json_from_string(response.content)
        
        llm_decision = json.loads(response_json)

        if llm_decision['next_action'] == 'STATE_TRANSITION':
            target_state = llm_decision.get('type')
            if target_state and self.state_machine_manager.can_transition_to(target_state):
                self.state_machine_manager.transition_to(target_state)
                print(f"State transition: {self.state_machine_manager.current_state} -> {target_state}")

                decision_type = NextActionDecisionType.GENERATE_ANSWER
                action = None

                next_action_decision = NextActionDecision(
                    type=decision_type,
                    action=action
                )
                return next_action_decision

        decision_type_mapping = {
            "GENERATE_ANSWER": NextActionDecisionType.GENERATE_ANSWER,
            "GUIDING_INSTRUCTIONS": NextActionDecisionType.GUIDING_INSTRUCTIONS,
            "ACTION": NextActionDecisionType.ACTION
        }

        decision_type = decision_type_mapping[llm_decision['next_action']]
        action = None
        if 'type' in llm_decision:
            action = llm_decision['type']

        next_action_decision = NextActionDecision(
            type=decision_type,
            action=action
        )

        print("LLM Decision Result:", next_action_decision)
        return next_action_decision
    
    def is_json_parsable(self, s):
        try:
            json.loads(s)
            return True
        except:
            print("Not JSON parsable")
            return False
        
    def extract_json_from_string(self, s):
        json_match = re.search(r'\{.*\}', s, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json_str
        return None

    def generate_dialog(self, chat_history_dict, instruction):
        dialog_output = ""
        for session_id, history in chat_history_dict.items():
            for message in history.messages:
                if isinstance(message, HumanMessage):
                    dialog_output += f"Mensch: {message.content}\n"
                elif isinstance(message, (AIMessage, AIMessageChunk)):
                    dialog_output += f"Chatbot: {message.content}\n"
                else:
                    dialog_output += f"Unbekannt: {message.content}\n"
        dialog_output += f"Mensch: {instruction}"
        return dialog_output.strip()