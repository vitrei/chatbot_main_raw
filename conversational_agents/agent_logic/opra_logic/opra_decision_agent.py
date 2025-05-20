from data_models.data_models import AgentState, NextActionDecision, NextActionDecisionType
from conversational_agents.agent_logic.base_decision_agent import BaseDecisionAgent

class OPRADecisionAgent(BaseDecisionAgent):

    def next_action(self, agent_state: AgentState):

        next_action_decision = NextActionDecision(
            type=NextActionDecisionType.GENERATE_ANSWER,
            action=None,
            payload=None
        )        

        if agent_state.conversation_turn_counter == 3:
            next_action_decision.type = NextActionDecisionType.GUIDING_INSTRUCTIONS
            next_action_decision.action = "location"
        
        elif agent_state.conversation_turn_counter == 4:
            next_action_decision.type = NextActionDecisionType.ACTION
            next_action_decision.action = "path_prediction"

        elif agent_state.conversation_turn_counter == 5:
            next_action_decision.type = NextActionDecisionType.ACTION
            next_action_decision.action = "parrot"

        else:
            next_action_decision.type = NextActionDecisionType.GUIDING_INSTRUCTIONS
            next_action_decision.action = "general_guidance"
        
        return next_action_decision 
        
        