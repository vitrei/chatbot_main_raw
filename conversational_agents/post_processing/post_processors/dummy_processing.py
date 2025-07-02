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
            
            # Add stage progression tracking with new state machine architecture
            if hasattr(agent_state, 'state_machine') and agent_state.state_machine:
                try:
                    state_machine = agent_state.state_machine
                    turn_counter = agent_state.conversation_turn_counter
                    
                    # Get current stage configuration
                    stage_config = state_machine.stages.get(state_machine.current_stage, {})
                    min_turns = stage_config.get('min_turns', 8)
                    max_turns = stage_config.get('max_turns', 12)
                    golden_path = stage_config.get('golden_path', [])
                    mandatory_sequence = stage_config.get('mandatory_sequence', [])
                    
                    # Calculate progress percentage
                    progress_percentage = min(100, (turn_counter / max_turns) * 100) if max_turns > 0 else 0
                    
                    # Get available transitions with guard rail info
                    available_transitions = state_machine.get_available_transitions(turn_counter)
                    allowed_transitions = [t for t in available_transitions if t.get('allowed', True)]
                    blocked_transitions = [t for t in available_transitions if not t.get('allowed', True)]
                    
                    # Check golden path position
                    golden_path_position = -1
                    if current_state in golden_path:
                        golden_path_position = golden_path.index(current_state)
                    
                    # Check mandatory sequence position  
                    sequence_position = -1
                    if current_state in mandatory_sequence:
                        sequence_position = mandatory_sequence.index(current_state)
                    
                    # Detect forced transitions
                    forced_transition = None
                    if turn_counter >= 12:
                        closure_transitions = [t for t in allowed_transitions if t['dest'] == 'onboarding_closure']
                        if closure_transitions:
                            forced_transition = closure_transitions[0]['trigger']
                        else:
                            emergency_transitions = [t for t in available_transitions if t['trigger'] == 'emergency_closure']
                            if emergency_transitions:
                                forced_transition = 'emergency_closure'
                    
                    # Add comprehensive progression data to payload
                    llm_answer.payload.update({
                        "stage_progression": {
                            "current_stage": current_stage,
                            "current_state": current_state,
                            "turn_counter": turn_counter,
                            "progress_percentage": round(progress_percentage, 1),
                            "min_turns": min_turns,
                            "max_turns": max_turns,
                            "turns_remaining": max(0, max_turns - turn_counter),
                            "golden_path_position": f"{golden_path_position + 1}/{len(golden_path)}" if golden_path_position >= 0 else "off-path",
                            "sequence_position": f"{sequence_position + 1}/{len(mandatory_sequence)}" if sequence_position >= 0 else "post-sequence",
                            "total_transitions": len(available_transitions),
                            "allowed_transitions": len(allowed_transitions),
                            "blocked_transitions": len(blocked_transitions),
                            "forced_transition": forced_transition,
                            "closure_approaching": turn_counter >= 11,
                            "mandatory_closure": turn_counter >= 12
                        }
                    })
                    
                except Exception as e:
                    print(f"❌ Error in stage progression tracking: {e}")
                    # Fallback minimal data
                    llm_answer.payload.update({
                        "stage_progression": {
                            "current_stage": current_stage,
                            "current_state": current_state,
                            "turn_counter": agent_state.conversation_turn_counter,
                            "error": str(e)
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
                    "max_turns": stage_data.get("max_turns", 12),
                    "min_turns": stage_data.get("min_turns", 8),
                    "turns_remaining": stage_data.get("turns_remaining", 0),
                    "golden_path_position": stage_data.get("golden_path_position", "unknown"),
                    "sequence_position": stage_data.get("sequence_position", "unknown"),
                    "total_transitions": stage_data.get("total_transitions", 0),
                    "allowed_transitions": stage_data.get("allowed_transitions", 0),
                    "blocked_transitions": stage_data.get("blocked_transitions", 0),
                    "forced_transition": stage_data.get("forced_transition"),
                    "closure_approaching": stage_data.get("closure_approaching", False),
                    "mandatory_closure": stage_data.get("mandatory_closure", False),
                    "progression_health": self._assess_progression_health(stage_data)
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
        """Assess if stage progression is healthy or problematic - UPDATED for new state machine"""
        turn_counter = stage_data.get("turn_counter", 0)
        progress_percentage = stage_data.get("progress_percentage", 0)
        max_turns = stage_data.get("max_turns", 12)
        current_state = stage_data.get("current_state", "unknown")
        current_stage = stage_data.get("current_stage", "unknown")
        golden_path_position = stage_data.get("golden_path_position", "unknown")
        forced_transition = stage_data.get("forced_transition")
        
        health_assessment = {
            "overall_health": "good",
            "issues": [],
            "recommendations": [],
            "turn_status": "normal"
        }
        
        # CRITICAL: Turn 12+ mandatory closure
        if turn_counter >= 12:
            if current_state != "onboarding_closure":
                health_assessment["overall_health"] = "critical"
                health_assessment["turn_status"] = "mandatory_closure"
                health_assessment["issues"].append("past_max_turns")
                health_assessment["recommendations"].append("immediate_closure_required")
            else:
                health_assessment["overall_health"] = "completed"
                health_assessment["turn_status"] = "successfully_closed"
        
        # WARNING: Turn 11 closure preparation
        elif turn_counter >= 11:
            health_assessment["turn_status"] = "closure_warning"
            if current_state in ["init_greeting", "engagement_hook", "stimulus_present"]:
                health_assessment["overall_health"] = "concerning"
                health_assessment["issues"].append("slow_progression_near_deadline")
                health_assessment["recommendations"].append("accelerate_to_closure")
            else:
                health_assessment["recommendations"].append("prepare_for_closure")
        
        # EARLY STAGES: Check for stagnation
        elif turn_counter > 5 and current_state == "init_greeting":
            health_assessment["overall_health"] = "concerning"
            health_assessment["issues"].append("stuck_in_greeting")
            health_assessment["recommendations"].append("force_progression_to_engagement")
        
        elif turn_counter > 7 and current_state in ["init_greeting", "engagement_hook"]:
            health_assessment["overall_health"] = "poor"
            health_assessment["issues"].append("slow_early_progression")
            health_assessment["recommendations"].append("stimulus_presentation_needed")
        
        # GOLDEN PATH: Check position
        if golden_path_position != "unknown" and golden_path_position != "off-path":
            if "1/" in golden_path_position or "2/" in golden_path_position:
                health_assessment["recommendations"].append("early_golden_path")
            elif "6/" in golden_path_position or "7/" in golden_path_position:
                health_assessment["recommendations"].append("approaching_closure")
        
        # FORCED TRANSITIONS: Check if system is forcing moves
        if forced_transition:
            health_assessment["turn_status"] = "forced_transition"
            health_assessment["recommendations"].append(f"system_forcing_{forced_transition}")
        
        # EXCELLENT: Good progression within time
        if (turn_counter <= 8 and 
            current_state in ["explore_path", "confirm_skepticism", "critical_thinking", "comfort_user"]):
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

