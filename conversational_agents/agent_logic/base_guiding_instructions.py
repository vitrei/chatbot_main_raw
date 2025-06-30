from abc import ABC, abstractmethod

from data_models.data_models import AgentState, NextActionDecision

class BaseGuidingInstructions(ABC):

    @abstractmethod
    def add_guiding_instructions(next_action:NextActionDecision, agent_state: AgentState) -> AgentState:
        pass


class GuidingInstructions(BaseGuidingInstructions):
    def __init__(self):
        super().__init__()

    def add_guiding_instructions(self, next_action: NextActionDecision, agent_state: AgentState) -> AgentState:
        guiding_instruction_name = next_action.action
        
        # Get behavioral guidance (young_user_guidance, gentle_approach, etc.)
        behavioral_instruction = self.get_behavioral_instruction(guiding_instruction_name, agent_state)
        
        # Get state-specific content guidance
        state_instruction = self.get_state_specific_instruction(agent_state)
        
        # Combine both instructions
        combined_instruction = self.combine_instructions(behavioral_instruction, state_instruction)
        
        if not hasattr(agent_state, 'current_guiding_instruction'):
            agent_state.current_guiding_instruction = ""
        
        agent_state.current_guiding_instruction = combined_instruction
        agent_state.current_guiding_instruction_name = guiding_instruction_name
        
        # Debug output
        current_state = "unknown"
        if hasattr(agent_state, 'state_machine') and agent_state.state_machine:
            current_state = agent_state.state_machine.get_current_state()
        
        print(f"✅ Added guiding instruction: {guiding_instruction_name}")
        print(f"🎰 Current State: {current_state}")
        print(f"📝 Behavioral Guidance: {behavioral_instruction[:100]}...")
        print(f"📋 State Content: {state_instruction[:100] if state_instruction else 'None'}...")
        
        return agent_state

    def get_behavioral_instruction(self, instruction_name, agent_state):
        """Get behavioral instruction (how the bot should behave)"""
        gi = agent_state.prompts.get('guiding_instructions', {})
        
        # Filter out state-based instructions, only return behavioral ones
        behavioral_instructions = {
            'general_guidance', 'young_user_guidance', 'expert_challenge', 
            'beginner_support', 'gentle_approach', 'quick_response'
        }
        
        if instruction_name in behavioral_instructions and instruction_name in gi:
            return gi[instruction_name]
        
        # Fallback to general guidance
        return gi.get('general_guidance', 'Continue the conversation naturally.')

    def get_state_specific_instruction(self, agent_state):
        """Get current state's content instruction (what to talk about)"""
        if not hasattr(agent_state, 'state_machine') or not agent_state.state_machine:
            return None
        
        current_state = agent_state.state_machine.get_current_state()
        state_prompts = agent_state.prompts.get('state_system_prompts', {})
        state_instructions = state_prompts.get(current_state, [])
        
        if state_instructions:
            return '\n'.join(state_instructions)
        
        return None

    def combine_instructions(self, behavioral_instruction, state_instruction):
        """Combine behavioral and state instructions"""
        if not state_instruction:
            return behavioral_instruction
        
        if not behavioral_instruction:
            return state_instruction
        
        return f"VERHALTEN: {behavioral_instruction}\n\nINHALT/PHASE: {state_instruction}"

    def get_instruction_content(self, instruction_name, agent_state):
        """
        Get instruction content, prioritizing state-specific instructions over general ones
        """
        # First try to get from general guiding instructions
        gi = agent_state.prompts.get('guiding_instructions', {})
        
        if instruction_name in gi:
            return gi[instruction_name]
        
        # If state machine is available, try to get state-specific instructions
        if hasattr(agent_state, 'state_machine') and agent_state.state_machine:
            current_state = agent_state.state_machine.get_current_state()
            
            # Check if this instruction name matches current state
            if instruction_name == current_state:
                state_prompts = agent_state.prompts.get('state_system_prompts', {})
                state_instructions = state_prompts.get(current_state, [])
                if state_instructions:
                    return '\n'.join(state_instructions)
        
        return None

    def get_combined_instruction(self, instruction_name, agent_state):
        """
        Combine state-specific instructions with general guiding instructions if both exist
        """
        general_instruction = agent_state.prompts.get('guiding_instructions', {}).get(instruction_name, '')
        
        if hasattr(agent_state, 'state_machine') and agent_state.state_machine:
            current_state = agent_state.state_machine.get_current_state()
            state_prompts = agent_state.prompts.get('state_system_prompts', {})
            state_instructions = state_prompts.get(current_state, [])
            
            if state_instructions:
                state_instruction_text = '\n'.join(state_instructions)
                
                if general_instruction:
                    return f"{state_instruction_text}\n\nZUSÄTZLICHE ANWEISUNGEN:\n{general_instruction}"
                else:
                    return state_instruction_text
        
        return general_instruction
# class GuidingInstructions(BaseGuidingInstructions):
    
#     def __init__(self):
#         super().__init__()

#     def add_guiding_instructions(self, next_action:NextActionDecision, agent_state: AgentState) -> AgentState:
#         gi = agent_state.prompts['guiding_instructions']

#         guiding_instruction_name = next_action.action

#         if guiding_instruction_name in gi:
#             agent_state.instruction += " " + gi[guiding_instruction_name]

#         return agent_state