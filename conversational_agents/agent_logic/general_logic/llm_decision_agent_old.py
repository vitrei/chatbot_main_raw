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

Der Chatbots soll nun die nächste sinnvolle Aktion ausführen. Mögliche Aktionen sind:
    GENERATE_ANSWER: Direkt eine Antwort generieren.
    GUIDING_INSTRUCTIONS: Den Dialog in eine bestimme Richtung lenken.
    ACTION: Eine externe Funktion aufrufen, z.B. einen API-Call.

Mögliche GUIDING_INSTRUCTIONS mit key und description sind:
    {guiding_instructions}

Mögliche ACTION mit key und description sind:
    {actions}

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
                ("system", "Du bist ein Decision Agent und legst für eine Konversation zwischen einem Chatbot und einem Menschen fest, welche Aktion der Chatbot als nächstes ausführen soll."),
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
            return ""
        except Exception as e:
            print(f"DEBUG: Could not get user profile from agent_state: {e}")
            return ""

    def format_user_profile_for_prompt(self, user_profile):
        """Format user profile data for the prompt"""
        if not user_profile:
            return ""
            
        profile_lines = ["BENUTZER-PROFIL:"]
        
        # Add available profile information
        if user_profile.get('age'):
            profile_lines.append(f"- Alter: {user_profile['age']}")
        if user_profile.get('gender'):
            profile_lines.append(f"- Geschlecht: {user_profile['gender']}")
        if user_profile.get('school_type'):
            profile_lines.append(f"- Schultyp: {user_profile['school_type']}")
        if user_profile.get('region'):
            profile_lines.append(f"- Region: {user_profile['region']}")
        if user_profile.get('social_media_usage'):
            profile_lines.append(f"- Social Media: {user_profile['social_media_usage']}")
        if user_profile.get('fake_news_skill'):
            profile_lines.append(f"- Fake News Kompetenz: {user_profile['fake_news_skill']}")
        if user_profile.get('fact_checking_habits'):
            profile_lines.append(f"- Faktenchecking: {user_profile['fact_checking_habits']}")
        if user_profile.get('vocabulary_level'):
            profile_lines.append(f"- Vokabular: {user_profile['vocabulary_level']}")
        if user_profile.get('interaction_style'):
            profile_lines.append(f"- Interaktionsstil: {user_profile['interaction_style']}")
        if user_profile.get('attention_span'):
            profile_lines.append(f"- Aufmerksamkeit: {user_profile['attention_span']}")
        if user_profile.get('current_mood'):
            profile_lines.append(f"- Stimmung: {user_profile['current_mood']}")
        if user_profile.get('interests'):
            interests_str = ", ".join(user_profile['interests'])
            profile_lines.append(f"- Interessen: {interests_str}")
        
        # Only add adaptive hints if we have meaningful profile data
        if len(profile_lines) > 1:  # More than just the header
            profile_lines.append("")
            profile_lines.append("ADAPTIVE HINWEISE:")
            
            age = user_profile.get('age')
            if age:
                if age < 16:
                    profile_lines.append("- Verwende sehr lockere, jugendliche Sprache")
                elif age < 18:
                    profile_lines.append("- Verwende lockere, altersgerechte Sprache")
            
            if user_profile.get('fake_news_skill') == 'master':
                profile_lines.append("- Benutzer hält sich für Fake News Experte - stelle vorsichtig in Frage")
            elif user_profile.get('fake_news_skill') == 'low':
                profile_lines.append("- Erkläre Fake News Konzepte einfach und verständlich")
            
            if user_profile.get('current_mood') == 'mad':
                profile_lines.append("- Benutzer ist schlecht gelaunt - sei besonders einfühlsam")
            
            if user_profile.get('attention_span') == 'short':
                profile_lines.append("- Halte Antworten extra kurz")
            
            return "\n".join(profile_lines)
        else:
            return ""

    def next_action(self, agent_state: AgentState):
        
        # DEBUG: Check what's in agent_state
        print(f"🔍 DEBUG Decision Agent:")
        print(f"   - agent_state type: {type(agent_state)}")
        print(f"   - agent_state user_id: {agent_state.user_id}")
        print(f"   - agent_state has user_profile attr: {hasattr(agent_state, 'user_profile')}")
        
        if hasattr(agent_state, 'user_profile'):
            print(f"   - agent_state.user_profile: {agent_state.user_profile}")
            print(f"   - user_profile type: {type(agent_state.user_profile)}")
        else:
            print(f"   - ❌ agent_state.user_profile MISSING!")
        
        # Get user profile from agent_state (populated by pre-processor)
        user_profile_info = self.get_user_profile_info(agent_state)
        
        prompts = prompt_loader.get_all_prompts()
        system_prompt = prompts['system_prompt']
        guiding_instruction_prompts = prompts['guiding_instructions']
        guidings_instructions_str = "" 
        for key, value in guiding_instruction_prompts.items():
            guidings_instructions_str += f"{key}: {value}\n"

        actions = """path_prediction: Empfehle einen Bildungspfad, wie die Person an den gewünschten Beruf kommt."""
        chat_history = self.generate_dialog(agent_state.chat_history, agent_state.instruction)
        
        print("User profile info:", user_profile_info if user_profile_info else "None available")
        print("chat_history", chat_history)

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

        print("next_action_decision:", next_action_decision)
        return next_action_decision
    
    # Keep all your existing helper methods unchanged
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