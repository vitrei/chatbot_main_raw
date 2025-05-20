from conversational_agents.agent_logic.base_conversational_agent_action_collection import BaseConversationalAgentActionsCollection
from conversational_agents.agent_logic.opra_logic.opra_actions.parrot_action import ParrotAction
from conversational_agents.agent_logic.opra_logic.opra_actions.path_recommendation_action import PathPredictionAction

class ConversationalAgentActionCollection(BaseConversationalAgentActionsCollection):

    def __init__(self):
        super().__init__()

        self.actions = {
            "path_prediction": PathPredictionAction(),
            "parrot": ParrotAction()
        }

    def get_action(self, action_name):
        if action_name not in self.actions:
            return None
        return self.actions[action_name]
    
    def get_actions(self):
        return self.actions

