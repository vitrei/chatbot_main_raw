from conversational_agents.post_processing.post_processors.base_post_processors import BasePostProcessor
import re
import unicodedata

class DummyProcessing(BasePostProcessor):

    def invoke(agent_state, llm_answer):
        emoji_pattern = re.compile(r"[\U00010000-\U0010FFFF]", flags=re.UNICODE)
        content = emoji_pattern.sub("", llm_answer.content)

        content = unicodedata.normalize('NFKD', content).encode('ascii', 'ignore').decode('ascii')

        content = content.replace('\n', ' ').replace('\r', '')

        llm_answer.content = content

        if llm_answer.payload is None:
            llm_answer.payload = {}

        llm_answer.payload["dummy_post_processing"] = {
            "foo": "bar"
        }

        return llm_answer

# class DummyProcessing(BasePostProcessor):

#     def invoke(agent_state, llm_answer):
#         # llm_answer.content += " ADDED_DUMMY"
#         llm_answer.content = llm_answer.content

#         if llm_answer.payload == None:
#             llm_answer.payload = {}

#         llm_answer.payload["dummy_post_processing"] = {
#             "foo": "bar"
#         }        
#         return llm_answer