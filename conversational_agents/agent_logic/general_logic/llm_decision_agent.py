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

Der Chatbots soll nun die nächste sinnvolle Aktion ausführen. Mögliche Aktionen sind:
    GENERATE: Direkt eine Antwort generieren.
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

    def next_action(self, agent_state: AgentState):
        
        prompts = prompt_loader.get_all_prompts()

        system_prompt = prompts['system_prompt']
        guiding_instruction_prompts = prompts['guiding_instructions']
        guidings_instructions_str = "" 
        for key, value in guiding_instruction_prompts.items():
            guidings_instructions_str += f"{key}: {value}\n"

        actions = """path_prediction: Empfehle einen Bildungspfad, wie die Person an den gewünschten Beruf kommt."""

        chat_history = self.generate_dialog(agent_state.chat_history, agent_state.instruction)
        
        print("chat_history", chat_history)

        response = self.chain.invoke(
            {
                "system_prompt": system_prompt,
                "chat_history": chat_history,
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
                    "chat_history": agent_state.chat_history,
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