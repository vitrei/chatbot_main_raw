from abc import ABC, abstractmethod


class BaseConversationalAgentActionsCollection(ABC):

    @abstractmethod
    def get_action(action_name:str):
        pass

    @abstractmethod
    def get_actions():
        pass