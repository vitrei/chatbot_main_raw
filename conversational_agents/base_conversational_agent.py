from abc import ABC, abstractmethod

class ConversationalAgent(ABC):

    @abstractmethod
    def proactive_instruct(self, instruction: str):
        pass

    @abstractmethod
    def proactive_stream(self, instruction: str):
        pass

    @abstractmethod
    def instruct(self, instruction: str):
        pass

    @abstractmethod
    def stream(self, instruction: str):
        pass