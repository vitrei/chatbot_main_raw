from data_models.data_models import AgentState, LLMAnswer


class ParrotAction:

    def __init__(self):
        pass

    def invoke(self, agent_state:AgentState) -> LLMAnswer:
        llm_answer = LLMAnswer(
            content=f"Deine Instruction war: {agent_state.instruction}",
            payload=None
        )

        return llm_answer