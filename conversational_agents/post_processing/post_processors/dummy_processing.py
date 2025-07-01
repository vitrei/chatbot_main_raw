import asyncio
import httpx
import datetime as dt
from conversational_agents.post_processing.post_processors.base_post_processors import BasePostProcessor
# from conversational_agents.agent_logic.general_logic.llm_decision_agent import LLMDecisionAgent
import re
import unicodedata

class DummyProcessing(BasePostProcessor):
    
    def __init__(self, user_profile_service_url: str = "http://localhost:8010", timeout: float = 2.0):
        self.user_profile_service_url = user_profile_service_url
        self.timeout = timeout
    
    def invoke(self, agent_state, llm_answer):
        emoji_pattern = re.compile(r"[\U00010000-\U0010FFFF]", flags=re.UNICODE)
        content = emoji_pattern.sub("", llm_answer.content)
        content = content.replace('\n', ' ').replace('\r', '')
        llm_answer.content = content
        
        if llm_answer.payload is None:
            current_state = "unknown"
            current_stage = "unknown"
            
            if hasattr(agent_state, 'state_machine') and agent_state.state_machine:
                current_state = agent_state.state_machine.get_current_state()
                current_stage = getattr(agent_state.state_machine, 'current_stage', 'unknown')
            
            llm_answer.payload = {"state": current_state, "stage": current_stage}
            
            # Add stage progression tracking
            if hasattr(agent_state, 'state_machine') and agent_state.state_machine:
                state_machine = agent_state.state_machine
                
                # Get stage context for progression tracking
                stage_context = state_machine.get_state_context_for_decision_agent(agent_state.conversation_turn_counter)
                stage_progress = stage_context.get('stage_progress', {})
                
                # Add comprehensive progression data to payload
                llm_answer.payload.update({
                    "stage_progression": {
                        "current_stage": current_stage,
                        "current_state": current_state,
                        "turn_counter": agent_state.conversation_turn_counter,
                        "progress_percentage": stage_progress.get('progress_percentage', 0),
                        "target_turns": stage_progress.get('target_turns', 15),
                        "milestones": stage_progress.get('milestone_status', []),
                        "stage_appropriate_transitions": len(stage_context.get('stage_appropriate_transitions', [])),
                        "total_transitions": len(stage_context.get('available_transitions', []))
                    }
                })
            
            # Check if fake news should be shown
            if (current_state == "stimulus_present" and 
                hasattr(agent_state, 'state_machine') and 
                agent_state.state_machine and
                hasattr(agent_state.state_machine, 'fake_news_stimulus_url') and
                agent_state.state_machine.fake_news_stimulus_url):
                
                fake_news_url = agent_state.state_machine.fake_news_stimulus_url
                llm_answer.payload["fake_news_url"] = fake_news_url
                
                # Optional: Füge URL zur Response hinzu
                llm_answer.content += f"\n\nSchau dir das mal an: {fake_news_url}"
                # print(f"🎬 Added fake news URL to response: {fake_news_url}")
        
        # if hasattr(agent_state, 'chat_history') and agent_state.chat_history:
        #     full_dialog = self.decision_agent.generate_dialog(agent_state.chat_history, "")
        #     print("-------------------")
        #     print(full_dialog)
        #     print("-------------------")

        #     chat_history = full_dialog.rstrip().rstrip("Mensch:").strip()
        #     llm_answer.payload["chat_history"] = chat_history
        # else:
        #     llm_answer.payload["chat_history"] = ""
        
        asyncio.create_task(self.send_conversation_async(agent_state, llm_answer))
        
        return llm_answer
    
    def create_conversation_summary(self, agent_state, llm_answer):

        try:
            full_chat_history = llm_answer.payload.get("chat_history", "")
            
            if not full_chat_history:
                return f"User: {agent_state.instruction or ''}\nBot: {llm_answer.content or ''}"
            
            messages = []
            for line in full_chat_history.split('\n'):
                line = line.strip()
                if line.startswith('Mensch: ') or line.startswith('User: '):
                    messages.append(('user', line.replace('Mensch: ', '').replace('User: ', '')))
                elif line.startswith('Chatbot: ') or line.startswith('Bot: '):
                    messages.append(('bot', line.replace('Chatbot: ', '').replace('Bot: ', '')))
            
            if agent_state.instruction:
                messages.append(('user', agent_state.instruction))
            if llm_answer.content:
                messages.append(('bot', llm_answer.content))
            
            relevant_messages = messages[-4:] if len(messages) >= 4 else messages
            
            summary_parts = []
            for msg_type, content in relevant_messages:
                if msg_type == 'user':
                    summary_parts.append(f"User: {content}")
                else:
                    summary_parts.append(f"Bot: {content}")
            
            summary = '\n'.join(summary_parts)
            
            context_hint = f"\n\nContext: Dies ist ein Gespräch über Fake News und Medienkompetenz. Der User ist {agent_state.user_profile.get('age', 'unbekanntes Alter')} Jahre alt."
            
            return summary + context_hint
            
        except Exception as e:
            print(f"Error creating conversation summary: {e}")
            return f"User: {agent_state.instruction or ''}\nBot: {llm_answer.content or ''}"

    async def send_conversation_async(self, agent_state, llm_answer):
        """Send conversation data to user profile builder (async, non-blocking)"""
        try:
            print(f"📡 Sending conversation for user {agent_state.user_id}")
            
            conversation_summary = self.create_conversation_summary(agent_state, llm_answer)

            # print(conversation_summary)
            
            # Extract stage progression data from payload
            stage_data = llm_answer.payload.get("stage_progression", {}) if llm_answer.payload else {}
            
            conversation_data = {
                "user_id": str(agent_state.user_id),
                "timestamp": dt.datetime.now().isoformat(),
                "user_message": agent_state.instruction or "",
                "bot_response": llm_answer.content,
                "full_conversation": conversation_summary,
                "turn_count": getattr(agent_state, 'conversation_turn_counter', 0),
                "user_profile": getattr(agent_state, 'user_profile', None),
                # Add stage progression tracking for user profile analysis
                "stage_progression": {
                    "current_stage": stage_data.get("current_stage", "unknown"),
                    "current_state": stage_data.get("current_state", "unknown"),
                    "progress_percentage": stage_data.get("progress_percentage", 0),
                    "turn_counter": stage_data.get("turn_counter", 0),
                    "target_turns": stage_data.get("target_turns", 15),
                    "milestones_status": stage_data.get("milestones", []),
                    "progression_health": self._assess_progression_health(stage_data),
                    "stage_transitions_available": stage_data.get("stage_appropriate_transitions", 0)
                }
            }
            
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{self.user_profile_service_url}/conversation"
                
                response = await client.post(url, json=conversation_data)
                
                # print(f"HTTP {response.status_code} from user profile service")
                
                if response.status_code == 200:
                    # print(f"Conversation sent successfully for user {agent_state.user_id}")
                    try:
                        response_json = response.json()
                        # print(f"Service response: {response_json}")
                    except:
                        print(f"Service response (text): {response.text[:200]}...")
                else:
                    print(f"User profile service returned {response.status_code}")
                    print(f"Response body: {response.text[:500]}...")
                    
        except httpx.TimeoutException as e:
            print(f"Timeout error sending conversation: {e}")
            print(f"   - Timeout was: {self.timeout}s")
            
        except httpx.ConnectError as e:
            print(f"Connection error sending conversation: {e}")
            print(f"   - Cannot connect to: {self.user_profile_service_url}")
            
        except Exception as e:
            print(f"Unexpected error sending conversation: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _assess_progression_health(self, stage_data) -> dict:
        """Assess if stage progression is healthy or problematic"""
        turn_counter = stage_data.get("turn_counter", 0)
        progress_percentage = stage_data.get("progress_percentage", 0)
        target_turns = stage_data.get("target_turns", 15)
        current_state = stage_data.get("current_state", "unknown")
        current_stage = stage_data.get("current_stage", "unknown")
        
        health_assessment = {
            "overall_health": "good",
            "issues": [],
            "recommendations": []
        }
        
        # Check for progression issues
        if turn_counter > 5 and current_state == "init_greeting":
            health_assessment["overall_health"] = "concerning"
            health_assessment["issues"].append("stuck_in_greeting")
            health_assessment["recommendations"].append("force_progression_to_engagement")
        
        if turn_counter > 10 and current_state in ["init_greeting", "engagement_hook"]:
            health_assessment["overall_health"] = "poor"
            health_assessment["issues"].append("slow_progression")
            health_assessment["recommendations"].append("stimulus_presentation_needed")
        
        if progress_percentage > 80 and current_state in ["init_greeting", "engagement_hook"]:
            health_assessment["overall_health"] = "critical"
            health_assessment["issues"].append("critical_stagnation")
            health_assessment["recommendations"].append("emergency_progression")
        
        # Check for good progression indicators
        if progress_percentage < 50 and current_state in ["stimulus_present", "reaction_wait"]:
            health_assessment["recommendations"].append("good_pacing")
        
        if turn_counter <= target_turns and current_state in ["explore_path", "confirm_skepticism", "meta_reflection"]:
            health_assessment["overall_health"] = "excellent"
            health_assessment["recommendations"].append("optimal_progression")
        
        return health_assessment

    


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

