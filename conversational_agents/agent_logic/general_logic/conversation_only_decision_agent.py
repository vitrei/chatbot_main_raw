from data_models.data_models import AgentState, NextActionDecision, NextActionDecisionType
from conversational_agents.agent_logic.base_decision_agent import BaseDecisionAgent

class ConversationOnlyDecisionAgent(BaseDecisionAgent):

    def __init__(self):
        super().__init__()

    def next_action(self, agent_state: AgentState):

        next_action_decision = NextActionDecision(
            type=NextActionDecisionType.GENERATE_ANSWER,
            action=None,
            payload=None
        )
                
        return next_action_decision