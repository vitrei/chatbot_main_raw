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
        # emoji processing
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
    
    def create_conversation_summary(self, agent_state, llm_answer):

        try:
            # Get full chat history
            full_chat_history = llm_answer.payload.get("chat_history", "")
            
            if not full_chat_history:
                # Fallback: create summary from current exchange only
                return f"User: {agent_state.instruction or ''}\nBot: {llm_answer.content or ''}"
            
            # Split into individual messages
            messages = []
            for line in full_chat_history.split('\n'):
                line = line.strip()
                if line.startswith('Mensch: ') or line.startswith('User: '):
                    messages.append(('user', line.replace('Mensch: ', '').replace('User: ', '')))
                elif line.startswith('Chatbot: ') or line.startswith('Bot: '):
                    messages.append(('bot', line.replace('Chatbot: ', '').replace('Bot: ', '')))
            
            # Add current exchange
            if agent_state.instruction:
                messages.append(('user', agent_state.instruction))
            if llm_answer.content:
                messages.append(('bot', llm_answer.content))
            
            # Get last 4 messages (2 user + 2 bot exchanges)
            relevant_messages = messages[-4:] if len(messages) >= 4 else messages
            
            # Create focused summary
            summary_parts = []
            for msg_type, content in relevant_messages:
                if msg_type == 'user':
                    summary_parts.append(f"User: {content}")
                else:
                    summary_parts.append(f"Bot: {content}")
            
            summary = '\n'.join(summary_parts)
            
            # Add context hint for better extraction
            context_hint = f"\n\nContext: Dies ist ein Gespräch über Fake News und Medienkompetenz. Der User ist {agent_state.user_profile.get('age', 'unbekanntes Alter')} Jahre alt."
            
            return summary + context_hint
            
        except Exception as e:
            print(f"Error creating conversation summary: {e}")
            # Fallback to current exchange
            return f"User: {agent_state.instruction or ''}\nBot: {llm_answer.content or ''}"

    async def send_conversation_async(self, agent_state, llm_answer):
        """Send conversation data to user profile builder (async, non-blocking)"""
        try:
            print(f"📡 Sending conversation for user {agent_state.user_id}")
            
            # Create focused conversation summary
            conversation_summary = self.create_conversation_summary(agent_state, llm_answer)
            
            # Prepare simple conversation data
            conversation_data = {
                "user_id": agent_state.user_id,
                "timestamp": dt.datetime.now().isoformat(),
                "user_message": agent_state.instruction or "",
                "bot_response": llm_answer.content,
                "full_conversation": conversation_summary,  # ← Changed to use summary
                "turn_count": getattr(agent_state, 'conversation_turn_counter', 0),
                "user_profile": getattr(agent_state, 'user_profile', None)
            }
            
            # Debug: Print data being sent
            # print(f"🔍 Conversation summary length: {len(conversation_summary)} chars")
            # print(f"🔍 Summary preview: {conversation_summary[:200]}...")
            # print(f"🔍 Target URL: {self.user_profile_service_url}/conversation")
            # print(f"🔍 Timeout: {self.timeout}s")
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{self.user_profile_service_url}/conversation"
                
                # Try the POST request
                response = await client.post(url, json=conversation_data)
                
                print(f"✅ HTTP {response.status_code} from user profile service")
                
                if response.status_code == 200:
                    print(f"✅ Conversation sent successfully for user {agent_state.user_id}")
                    try:
                        response_json = response.json()
                        print(f"📝 Service response: {response_json}")
                    except:
                        print(f"📝 Service response (text): {response.text[:200]}...")
                else:
                    print(f"❌ User profile service returned {response.status_code}")
                    print(f"📝 Response body: {response.text[:500]}...")
                    
        except httpx.TimeoutException as e:
            print(f"⏰ Timeout error sending conversation: {e}")
            print(f"   - Timeout was: {self.timeout}s")
            
        except httpx.ConnectError as e:
            print(f"🔌 Connection error sending conversation: {e}")
            print(f"   - Cannot connect to: {self.user_profile_service_url}")
            
        except Exception as e:
            print(f"❌ Unexpected error sending conversation: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()

    


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

