import requests
from typing import Dict, Any, Optional
from conversational_agents.pre_processing.pre_processors.base_pre_processors import BasePreProcessor
from data_models.data_models import AgentState

import asyncio
import httpx
from typing import Dict, Any

class FakeNewsPreProcessor(BasePreProcessor):
    def __init__(self, file_server_url: str = "http://localhost:8000", timeout: float = 3.0):
        self.file_server_url = file_server_url
        self.target_face_path = "/home/merlotllm/Documents/facefusion/temp/b8ce6513-2ffd-4823-8bc5-3058abc656cb_source.jpg"
        self.timeout = timeout
        self._fetched_content = {}
        print(f"FakeNewsPreProcessor initialized with server: {file_server_url}")
    
    def invoke(self, agent_state):
        """
        Non-blocking invoke - starts async file check in background
        """
        print(f"🔍 Starting fake news availability check for user {agent_state.user_id}")
        
        # Start async file check (non-blocking)
        asyncio.create_task(self.check_and_process_files_async(agent_state))
        
        # Return immediately - file check happens in background
        return agent_state
    
    async def check_and_process_files_async(self, agent_state):
        """
        Check file availability and trigger faceswap if needed (async)
        """
        try:
            result = await self.check_fake_news_availability_async(agent_state.user_id)
            
            # Store result in agent_state for future use
            if hasattr(agent_state, 'fake_news_files'):
                agent_state.fake_news_files = result
            else:
                setattr(agent_state, 'fake_news_files', result)
                
            print(f"✅ File availability check complete for user {agent_state.user_id}")
            
        except Exception as e:
            print(f"❌ Error in async file processing: {e}")
    
    async def check_fake_news_availability_async(self, user_id: str) -> Dict[str, Any]:
        """Check if fake news files are available for the user (async)"""
        try:
            url = f"{self.file_server_url}/check-file/{user_id}"
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"File availability: {result}")
                    
                    # If JPG doesn't exist, trigger faceswap asynchronously
                    if not result.get("jpg_exists", False):
                        print(f"JPG missing for user {user_id} - triggering faceswap...")
                        # Fire and forget - don't wait for faceswap to complete
                        asyncio.create_task(self.trigger_faceswap_async(user_id))
                    
                    return result
                else:
                    print(f"File check failed with status {response.status_code}")
                    return {"jpg_exists": False, "mp4_exists": False}
                    
        except httpx.TimeoutException:
            print(f"❌ Timeout checking files for user {user_id}")
            return {"jpg_exists": False, "mp4_exists": False}
        except httpx.ConnectError:
            print(f"❌ Connection error checking files for user {user_id}")
            return {"jpg_exists": False, "mp4_exists": False}
        except Exception as e:
            print(f"❌ Error checking fake news files: {e}")
            return {"jpg_exists": False, "mp4_exists": False}
    
    async def trigger_faceswap_async(self, user_id: str):
        """Trigger faceswap API when JPG is missing (async, fire-and-forget)"""
        try:
            faceswap_url = f"{self.file_server_url}/faceswap"
            payload = {
                "user_id": user_id,
                "target_face_path": self.target_face_path
            }
            
            print(f"🎭 POST {faceswap_url}")
            print(f"Payload: {payload}")
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(faceswap_url, json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ Faceswap successful for user {user_id}")
                    print(f"Response: {result}")
                else:
                    print(f"❌ Faceswap failed: HTTP {response.status_code}")
                    print(f"Response: {response.text}")
                    
        except httpx.TimeoutException:
            print(f"❌ Faceswap timeout for user {user_id}")
        except httpx.ConnectError:  
            print(f"❌ Faceswap connection error for user {user_id}")
        except Exception as e:
            print(f"❌ Error triggering faceswap: {e}")