from dataclasses import dataclass
from enum import auto
from typing import List

class NextActionDecisionType:
    GENERATE_ANSWER = auto()
    PROMPT_ADAPTION = auto()
    GUIDING_INSTRUCTIONS = auto()
    ACTION = auto()

@dataclass
class NextActionDecision:
    type: NextActionDecisionType
    action: str
    payload: dict | None = None

@dataclass
class RAGDocument:
    content: str
    metadata: dict | None = None

@dataclass
class LLMAnswer:
    content: str
    payload: dict | None = None
    rag_context: List[RAGDocument] | None = None

@dataclass
class AgentState:
    user_id:str
    conversation_turn_counter: int
    instruction: str
    chat_history: list
    prompts: dict
    user_profile: dict = None
    

