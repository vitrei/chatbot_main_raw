import asyncio
import httpx
import datetime as dt
from conversational_agents.post_processing.post_processors.base_post_processors import BasePostProcessor
from conversational_agents.agent_logic.general_logic.llm_decision_agent import LLMDecisionAgent
import re
import unicodedata

class DummyProcessing(BasePostProcessor):
    
    def __init__(self, user_profile_service_url: str = "http://localhost:8010", timeout: float = 2.0):
        self.decision_agent = LLMDecisionAgent()
        self.user_profile_service_url = user_profile_service_url
        self.timeout = timeout
    
    def invoke(self, agent_state, llm_answer):
        # Existing emoji processing
        emoji_pattern = re.compile(r"[\U00010000-\U0010FFFF]", flags=re.UNICODE)
        content = emoji_pattern.sub("", llm_answer.content)
        content = content.replace('\n', ' ').replace('\r', '')
        llm_answer.content = content
        
        if llm_answer.payload is None:
            llm_answer.payload = {}
        
        # Existing chat history processing
        if hasattr(agent_state, 'chat_history') and agent_state.chat_history:
            full_dialog = self.decision_agent.generate_dialog(agent_state.chat_history, "")
            chat_history = full_dialog.rstrip().rstrip("Mensch:").strip()
            llm_answer.payload["chat_history"] = chat_history
        else:
            llm_answer.payload["chat_history"] = ""
        
        # NEW: Send conversation data asynchronously (fire-and-forget)
        asyncio.create_task(self.send_conversation_async(agent_state, llm_answer))
        
        return llm_answer
    
    async def send_conversation_async(self, agent_state, llm_answer):
        """Send conversation data to user profile builder (async, non-blocking)"""
        try:
            print(f"📡 Sending conversation for user {agent_state.user_id}")
            
            # Prepare simple conversation data
            conversation_data = {
                "user_id": agent_state.user_id,
                "timestamp": dt.datetime.now().isoformat(),
                "user_message": agent_state.instruction or "",
                "bot_response": llm_answer.content,
                "full_conversation": llm_answer.payload.get("chat_history", ""),
                "turn_count": getattr(agent_state, 'conversation_turn_counter', 0),
                "user_profile": getattr(agent_state, 'user_profile', None)
            }
            
            # Send async POST
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{self.user_profile_service_url}/conversation"
                response = await client.post(url, json=conversation_data)
                
                if response.status_code == 200:
                    print(f"✅ Conversation sent for user {agent_state.user_id}")
                else:
                    print(f"⚠️  User profile service returned {response.status_code}")
                    
        except Exception as e:
            print(f"❌ Error sending conversation: {e}")


# class DummyProcessing(BasePostProcessor):


#     def __init__(self):
#         self.decision_agent = LLMDecisionAgent()

#     def invoke(self, agent_state, llm_answer):
#         emoji_pattern = re.compile(r"[\U00010000-\U0010FFFF]", flags=re.UNICODE)
#         content = emoji_pattern.sub("", llm_answer.content)

#         # content = unicodedata.normalize('NFKD', content).encode('ascii', 'ignore').decode('ascii')

#         content = content.replace('\n', ' ').replace('\r', '')

#         llm_answer.content = content

#         if llm_answer.payload is None:
#             llm_answer.payload = {}

#         # llm_answer.payload["dummy_post_processing"] = {
#         #     "foo": "bar"
#         # }

#         if hasattr(agent_state, 'chat_history') and agent_state.chat_history:
#             full_dialog = self.decision_agent.generate_dialog(agent_state.chat_history, "")
#             chat_history = full_dialog.rstrip().rstrip("Mensch:").strip()
#             llm_answer.payload["chat_history"] = chat_history
#         else:
#             llm_answer.payload["chat_history"] = ""

#         return llm_answer

