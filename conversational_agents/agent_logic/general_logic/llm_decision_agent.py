import json
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage
from langchain_core.messages.ai import AIMessageChunk

from data_models.data_models import AgentState, NextActionDecision, NextActionDecisionType
from conversational_agents.agent_logic.base_decision_agent import BaseDecisionAgent
from large_language_models.llm_factory import llm_factory
from prompts.prompt_loader import prompt_loader

class LLMDecisionAgent(BaseDecisionAgent):

    def __init__(self):
        super().__init__()
        
        decision_agent_prompt = """"
Der Chatbot ist definiert durch folgenden Prompt:
{system_prompt}

Das ist der Dialog zwischen dem Chatbot und einem Menschen:
{chat_history}

{user_profile_info}

WICHTIG: Berücksichtige das Benutzerprofil bei der Entscheidung! Wähle die Aktion, die am besten zum Benutzer passt.

Der Chatbots soll nun die nächste sinnvolle Aktion ausführen. Mögliche Aktionen sind:
    GENERATE_ANSWER: Direkt eine Antwort generieren.
    GUIDING_INSTRUCTIONS: Den Dialog in eine bestimme Richtung lenken.
    ACTION: Eine externe Funktion aufrufen, z.B. einen API-Call.

Mögliche GUIDING_INSTRUCTIONS mit key und description sind:
    {guiding_instructions}

Mögliche ACTION mit key und description sind:
    {actions}

ENTSCHEIDUNGSHILFEN basierend auf Benutzerprofil:
- Wenn Alter < 16: Nutze "young_user_guidance" 
- Wenn fake_news_skill = "master": Nutze "expert_challenge"
- Wenn fake_news_skill = "low": Nutze "beginner_support" 
- Wenn current_mood = "mad": Nutze "gentle_approach"
- Wenn attention_span = "short": Nutze "quick_response"
- Bei ersten Gesprächen: Mehr "source_check" und "skepticism"
- Bei späteren Gesprächen: Mehr "emotional_content" und fortgeschrittene Techniken

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

    def next_action(self, agent_state: AgentState):
        
        # DEBUG: Check what's in agent_state
        # print(f"DEBUG LLM Decision Agent:")
        # print(f"   - agent_state type: {type(agent_state)}")
        # print(f"   - agent_state user_id: {agent_state.user_id}")
        # print(f"   - conversation_turn_counter: {agent_state.conversation_turn_counter}")
        # print(f"   - agent_state has user_profile attr: {hasattr(agent_state, 'user_profile')}")
        
        
        user_profile_info = self.get_user_profile_info(agent_state)
        
        prompts = prompt_loader.get_all_prompts()
        system_prompt = prompts['system_prompt']
        guiding_instruction_prompts = prompts['guiding_instructions']
        guidings_instructions_str = "" 
        for key, value in guiding_instruction_prompts.items():
            guidings_instructions_str += f"{key}: {value}\n"

        actions = """Keine spezifischen Actions definiert für Fake News Gespräche."""
        chat_history = self.generate_dialog(agent_state.chat_history, agent_state.instruction)
        
        print("🔍 User profile info for LLM:", user_profile_info if user_profile_info else "None available")
        print("🔍 Chat history:", chat_history)
        print(f"🔍 Turn counter: {agent_state.conversation_turn_counter}")

        response = self.chain.invoke(
            {
                "system_prompt": system_prompt,
                "chat_history": chat_history,
                "user_profile_info": user_profile_info,
                "guiding_instructions": guidings_instructions_str,
                "actions": actions
            }
        )

        response_json = response.content

        while response_json == None or not self.is_json_parsable(response_json):
            print("Not a valid JSON. Retrying...")
            response = self.chain.invoke(
                {
                    "system_prompt": system_prompt,
                    "chat_history": chat_history,
                    "user_profile_info": user_profile_info,
                    "guiding_instructions": guidings_instructions_str,
                    "actions": actions
                }
            )
            response_json = self.extract_json_from_string(response.content)
        
        llm_decision = json.loads(response_json)

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

        print("🎯 LLM Decision Result:", next_action_decision)
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