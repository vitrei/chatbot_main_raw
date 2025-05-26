from conversational_agents.post_processing.post_processors.base_post_processors import BasePostProcessor


class DummyProcessing(BasePostProcessor):

    def invoke(agent_state, llm_answer):
        # llm_answer.content += " ADDED_DUMMY"
        llm_answer.content = llm_answer.content

        if llm_answer.payload == None:
            llm_answer.payload = {}

        llm_answer.payload["dummy_post_processing"] = {
            "foo": "bar"
        }        
        return llm_answer