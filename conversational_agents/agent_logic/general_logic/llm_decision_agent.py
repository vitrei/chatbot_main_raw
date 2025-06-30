import json
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage
from langchain_core.messages.ai import AIMessageChunk

from data_models.data_models import AgentState, NextActionDecision, NextActionDecisionType
from conversational_agents.agent_logic.base_decision_agent import BaseDecisionAgent
from large_language_models.llm_factory import llm_factory
from conversational_agents.agent_logic.state_machine_wrapper import create_state_machine_from_prompts
from prompts.prompt_loader import prompt_loader

class LLMDecisionAgent(BaseDecisionAgent):

    def __init__(self):
        super().__init__()
        
        decision_agent_prompt = """
Der Chatbot ist definiert durch folgenden Prompt:
{system_prompt}

AKTUELLER GESPRÄCHSVERLAUF:
{chat_history}

STATE MACHINE KONTEXT:
- Aktueller State: {current_state}
- Mögliche Transitions: {available_transitions}
- State-spezifische Anweisungen: {state_specific_instructions}

GESPRÄCHS-KONTEXT:
- Turn: {turn_counter}
- Letzte User-Nachricht: {last_user_message}

BENUTZERPROFIL:
{user_profile_info}

WICHTIG: Du steuerst eine strukturierte Fake-News-Aufklärungs-Konversation durch State Machine Phasen!

DEINE AUFGABEN:
1. ANALYSIERE die User-Reaktion und den aktuellen State
2. WÄHLE die passende Guiding Instruction für das VERHALTEN des Bots (young_user_guidance, gentle_approach, etc.)
3. ENTSCHEIDE ob ein State-Transition notwendig ist basierend auf dem INHALT/THEMA

WICHTIGE UNTERSCHEIDUNG:
- GUIDING_INSTRUCTIONS = WIE der Bot sich verhält (Tonfall, Stil, Länge)
- STATE_TRANSITIONS = WAS der Bot thematisch behandeln soll (init_greeting, stimulus_present, etc.)

ENTSCHEIDUNGSLOGIK FÜR GUIDING INSTRUCTIONS:
- Alter <16: "young_user_guidance" 
- User emotional aufgewühlt: "gentle_approach"
- User braucht kurze Antworten: "quick_response"
- User ist Experte: "expert_challenge"
- User ist Anfänger: "beginner_support"
- Sonst: "general_guidance"

ENTSCHEIDUNGSLOGIK FÜR STATE TRANSITIONS:
- Turn 0-1: Von init_greeting zu engagement_hook
- User zeigt Interesse: Weiter zur nächsten Phase
- User skeptisch/ablehnend: conversation_repair
- User emotional: comfort_needed
- Phase abgeschlossen: Zur logisch nächsten Phase

VERFÜGBARE GUIDING_INSTRUCTIONS (für BOT-VERHALTEN):
{guiding_instructions}

MÖGLICHE TRANSITIONS (für THEMEN-WECHSEL):
{transition_options}

Du gibst deine Antwort als JSON zurück:

{{
    "next_action": "GUIDING_INSTRUCTIONS",
    "guiding_instruction": "<behavioral_instruction_key>",
    "state_transition": {{
        "needed": true/false,
        "trigger": "<trigger_name>",
        "reason": "<warum dieser transition>"
    }},
    "reason": "<kurze Begründung für Instruction-Wahl>"
}}
"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Du bist ein intelligenter Decision Agent für strukturierte Fake-News-Aufklärungsgespräche mit State Machine Integration."),
            ("human", decision_agent_prompt),
        ])

        llm = llm_factory.get_llm()
        self.chain = prompt | llm 

    def get_state_machine_context(self, agent_state):
        """Extract state machine context from agent state"""
        if not hasattr(agent_state, 'state_machine') or not agent_state.state_machine:
            print("❌ No state machine found in agent_state")
            return {
                'current_state': 'init_greeting',  # Use default initial state
                'available_transitions': [],
                'state_specific_instructions': 'No state machine available - using default state'
            }
        
        sm = agent_state.state_machine
        context = sm.get_state_context_for_decision_agent()
        print(f"🎰 State Machine Context: {context}")
        
        # Get state-specific instructions from prompts
        state_prompts = agent_state.prompts.get('state_system_prompts', {})
        current_state_instructions = state_prompts.get(context['current_state'], ['No specific instructions'])
        
        return {
            'current_state': context['current_state'],
            'available_transitions': context['available_transitions'],
            'state_specific_instructions': '\n'.join(current_state_instructions)
        }

    def get_transition_options_text(self, available_transitions):
        """Format available transitions for prompt"""
        if not available_transitions:
            return "Keine Transitions verfügbar"
        
        options = []
        for transition in available_transitions:
            options.append(f"- {transition['trigger']}: {transition['description']} -> {transition['dest']}")
        
        return '\n'.join(options)

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

    def next_action(self, agent_state: AgentState):
        print(f"🔢 TURN COUNTER: {agent_state.conversation_turn_counter}")
        
        # Ensure state machine is initialized
        agent_state = self.add_state_machine_to_agent_state(agent_state)
        
        # Get state machine context
        sm_context = self.get_state_machine_context(agent_state)
        print(f"🎰 CURRENT STATE: {sm_context['current_state']}")
        print(f"🔄 AVAILABLE TRANSITIONS: {len(sm_context['available_transitions'])}")
        
        user_profile_info = self.get_user_profile_info(agent_state)
        
        prompts = prompt_loader.get_all_prompts()
        system_prompt = prompts['system_prompt']
        guiding_instruction_prompts = prompts['guiding_instructions']
        
        # Format guiding instructions for prompt
        guidings_instructions_str = ""
        for key, value in guiding_instruction_prompts.items():
            guidings_instructions_str += f"{key}: {value}\n"

        chat_history = self.generate_dialog(agent_state.chat_history, agent_state.instruction)
        last_user_message = self.get_last_user_message(agent_state.chat_history)
        transition_options = self.get_transition_options_text(sm_context['available_transitions'])
        
        prompt_data = {
            "system_prompt": system_prompt,
            "chat_history": chat_history,
            "current_state": sm_context['current_state'],
            "available_transitions": sm_context['available_transitions'],
            "state_specific_instructions": sm_context['state_specific_instructions'],
            "turn_counter": agent_state.conversation_turn_counter,
            "last_user_message": last_user_message,
            "user_profile_info": user_profile_info,
            "guiding_instructions": guidings_instructions_str,
            "transition_options": transition_options
        }
        
        response = self.chain.invoke(prompt_data)
        response_json = response.content

        # Retry logic for invalid JSON
        retry_count = 0
        max_retries = 3
        while (response_json == None or not self.is_json_parsable(response_json)) and retry_count < max_retries:
            print(f"Not a valid JSON. Retrying... ({retry_count + 1}/{max_retries})")
            response = self.chain.invoke(prompt_data)
            response_json = self.extract_json_from_string(response.content)
            retry_count += 1
        
        if not self.is_json_parsable(response_json):
            print("❌ Failed to get valid JSON after retries. Using fallback decision.")
            return self.create_fallback_decision(sm_context)
        
        llm_decision = json.loads(response_json)
        print(f"🤖 LLM DECISION RAW: {llm_decision}")

        # Handle state transition if needed
        if llm_decision.get('state_transition', {}).get('needed', False):
            transition_info = llm_decision['state_transition']
            trigger = transition_info.get('trigger')
            reason = transition_info.get('reason', 'No reason provided')
            
            if hasattr(agent_state, 'state_machine') and agent_state.state_machine:
                success = agent_state.state_machine.execute_transition(trigger, reason)
                if success:
                    print(f"✅ STATE TRANSITION EXECUTED: {trigger}")
                else:
                    print(f"❌ STATE TRANSITION FAILED: {trigger}")

        # Create decision for guiding instruction
        guiding_instruction = llm_decision.get('guiding_instruction', sm_context['current_state'])
        
        if 'reason' in llm_decision:
            print(f"📝 INSTRUCTION REASON: {llm_decision['reason']}")

        next_action_decision = NextActionDecision(
            type=NextActionDecisionType.GUIDING_INSTRUCTIONS,
            action=guiding_instruction
        )

        print(f"✅ FINAL DECISION: {next_action_decision}")
        return next_action_decision

    def create_fallback_decision(self, sm_context):
        """Create a fallback decision when LLM fails"""
        current_state = sm_context['current_state']
        
        # Use current state as guiding instruction if available
        fallback_instruction = current_state if current_state != 'unknown' else 'general_guidance'
        
        return NextActionDecision(
            type=NextActionDecisionType.GUIDING_INSTRUCTIONS,
            action=fallback_instruction
        )
    
    def is_json_parsable(self, s):
        try:
            json.loads(s)
            return True
        except:
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