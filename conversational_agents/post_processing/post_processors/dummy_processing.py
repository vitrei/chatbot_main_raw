from conversational_agents.post_processing.post_processors.base_post_processors import BasePostProcessor
from conversational_agents.agent_logic.general_logic.llm_decision_agent import LLMDecisionAgent
import httpx
import datetime as dt
import asyncio
import re
import unicodedata


class DummyProcessing(BasePostProcessor):

    def __init__(self, user_profile_service_url: str = "http://localhost:8010", timeout: float = 2.0):
        self.decision_agent = LLMDecisionAgent()
        self.user_profile_service_url = user_profile_service_url
        self.timeout = timeout

    def invoke(self, agent_state, llm_answer):
        # Remove emojis from the response content
        emoji_pattern = re.compile(r"[\U00010000-\U0010FFFF]", flags=re.UNICODE)
        content = emoji_pattern.sub("", llm_answer.content)
        content = content.replace('\n', ' ').replace('\r', '')
        llm_answer.content = content

        if llm_answer.payload is None:
            llm_answer.payload = {}

        if hasattr(agent_state, 'chat_history') and agent_state.chat_history:
            full_dialog = self.decision_agent.generate_dialog(agent_state.chat_history, "")
            chat_history = full_dialog.rstrip().rstrip("Mensch:").strip()
            llm_answer.payload["chat_history"] = chat_history
        else:
            llm_answer.payload["chat_history"] = ""

        asyncio.create_task(self.send_conversation_async(agent_state, llm_answer))

        return llm_answer
            
    async def send_conversation_async(self, agent_state, llm_answer):
        """Send conversation data to user profile builder asynchronously"""
        try:
            print(f"🔄 Sending conversation for user {agent_state.user_id}")
            
            # Get only last 2 messages instead of full history
            recent_conversation = ""
            if hasattr(agent_state, 'chat_history') and agent_state.chat_history:
                print(f"📊 Chat history length: {len(agent_state.chat_history)}")
                # Get the last 2 exchanges (user + bot messages)
                chat_messages = agent_state.chat_history[-4:] if len(agent_state.chat_history) >= 4 else agent_state.chat_history
                print(f"📝 Using last {len(chat_messages)} messages")
                
                try:
                    recent_dialog = self.decision_agent.generate_dialog(chat_messages, "")
                    recent_conversation = recent_dialog.rstrip().rstrip("Mensch:").strip()
                    print(f"✅ Generated dialog: {recent_conversation[:100]}...")
                except Exception as dialog_error:
                    print(f"❌ Error generating dialog: {dialog_error}")
                    recent_conversation = ""
            else:
                print("ℹ️ No chat history available")
            
            conversation_data = {
                "user_id": agent_state.user_id,
                "timestamp": dt.datetime.now().isoformat(),
                "user_message": agent_state.instruction or "",
                "bot_response": llm_answer.content,
                "full_conversation": recent_conversation,
                "turn_count": getattr(agent_state, 'conversation_turn_counter', 0),
                "user_profile": getattr(agent_state, 'user_profile', None)
            }
            
            print(f"📦 Conversation data prepared:")
            print(f"  - User ID: {conversation_data['user_id']}")
            print(f"  - User message: {conversation_data['user_message'][:50]}...")
            print(f"  - Bot response: {conversation_data['bot_response'][:50]}...")
            
            url = f"{self.user_profile_service_url}/conversation"
            print(f"🌐 Sending POST to: {url}")
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=conversation_data)
                
                print(f"📡 Response status: {response.status_code}")
                
                if response.status_code == 200:
                    print(f"✅ Conversation sent for user {agent_state.user_id}")
                    try:
                        response_json = response.json()
                        print(f"📄 Response: {response_json}")
                    except Exception:
                        print("📄 Response text:", response.text[:200])
                else:
                    print(f"❌ User profile service returned {response.status_code}")
                    print(f"📄 Error response: {response.text}")
                    
        except httpx.TimeoutException as e:
            print(f"⏰ Timeout error sending conversation: {e}")
        except httpx.RequestError as e:
            print(f"🌐 Network error sending conversation: {e}")
        except Exception as e:
            print(f"❌ Unexpected error sending conversation: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    # async def send_conversation_async(self, agent_state, llm_answer):
    #     """Send conversation data to user profile builder asynchronously"""
    #     try:
    #         print(f"Sending conversation for user {agent_state.user_id}")

    #         conversation_data = {
    #             "user_id": agent_state.user_id,
    #             "timestamp": dt.datetime.now().isoformat(),
    #             "user_message": agent_state.instruction or "",
    #             "bot_response": llm_answer.content,
    #             "full_conversation": llm_answer.payload.get("chat_history", ""),
    #             "turn_count": getattr(agent_state, 'conversation_turn_counter', 0),
    #             "user_profile": getattr(agent_state, 'user_profile', None)
    #         }

    #         async with httpx.AsyncClient(timeout=self.timeout) as client:
    #             url = f"{self.user_profile_service_url}/conversation"
    #             response = await client.post(url, json=conversation_data)

    #             if response.status_code == 200:
    #                 print(f"Conversation sent for user {agent_state.user_id}")
    #             else:
    #                 print(f"User profile service returned {response.status_code}")

    #     except Exception as e:
    #         print(f"Error sending conversation: {e}")



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