from data_models.data_models import AgentState, LLMAnswer


class PathPredictionAction:

    def __init__(self):
        pass

    def invoke(self, agent_state:AgentState) -> LLMAnswer:
        llm_answer = LLMAnswer(
            content="Du musst Informatik an der HKA studieren!",
            payload={
                "type":"educationalPath", 
                "data": {
                    "title": "Informatik",
                    "description": "Lorem Ipsum"
                }
            }
        )

        return llm_answer