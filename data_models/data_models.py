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
    user_id: str | int
    conversation_turn_counter: int
    instruction: str
    chat_history: list
    prompts: dict
    user_profile: dict = None
    state_machine: 'ConversationStateMachine' = None  # Add state machine reference
    current_guiding_instruction: str = None  # Current guiding instruction content
    current_guiding_instruction_name: str = None  # Current guiding instruction name
    current_state_prompts: list = None  # State-specific system prompts from state machine
    current_state_examples: list = None  # State-specific examples from state machine

