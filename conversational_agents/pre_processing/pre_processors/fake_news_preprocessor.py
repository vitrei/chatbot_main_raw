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
        self.target_video_path = "/home/merlotllm/Documents/facefusion/temp/b8ce6513-2ffd-4823-8bc5-3058abc656cb_target.mp4"
        self.timeout = timeout
        self._fetched_content = {}
        self._fake_news_available = {}
        # print(f"FakeNewsPreProcessor initialized with server: {file_server_url}")
    
    def invoke(self, agent_state):
        """
        Non-blocking invoke - starts async file check in background
        """
        print("=== FAKE NEWS PREPROCESSOR ===")
        print(f"Checking availability for user: {agent_state.user_id}")
        
        # Check cache first
        if agent_state.user_id in self._fake_news_available:
            fake_news_ready = self._fake_news_available[agent_state.user_id]
            print(f"📋 Cached result: Fake news available = {fake_news_ready}")
            
            if fake_news_ready and hasattr(agent_state, 'state_machine') and agent_state.state_machine:
                agent_state.state_machine.set_fake_news_stimulus_available(agent_state.user_id)
        else:
            # Start async check in background
            asyncio.create_task(self.check_and_process_files_async(agent_state))
        
        return agent_state
    
    async def check_fake_news_availability_async(self, user_id: str) -> Dict[str, Any]:
        """Check if fake news files are available for the user (async)"""
        try:
            url = f"{self.file_server_url}/check-file/{user_id}"
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"File availability: {result}")
                    
                    jpg_missing = not result.get("jpg_exists", False)
                    mp4_missing = not result.get("mp4_exists", False)
                    
                    if jpg_missing or mp4_missing:
                        asyncio.create_task(self.process_missing_files_sequentially(user_id, jpg_missing, mp4_missing))
                    
                    return result
                else:
                    print(f"File check failed with status {response.status_code}")
                    return {"jpg_exists": False, "mp4_exists": False}
                    
        except httpx.TimeoutException:
            print(f"Timeout checking files for user {user_id}")
            return {"jpg_exists": False, "mp4_exists": False}
        except httpx.ConnectError:
            print(f"Connection error checking files for user {user_id}")
            return {"jpg_exists": False, "mp4_exists": False}
        except Exception as e:
            print(f"Error checking fake news files: {e}")
            return {"jpg_exists": False, "mp4_exists": False}

    async def check_and_process_files_async(self, agent_state):
        """
        Check file availability and cache result
        """
        try:
            result = await self.check_fake_news_availability_async(agent_state.user_id)
            
            # Cache the result
            fake_news_ready = result.get('jpg_exists', False) and result.get('mp4_exists', False)
            self._fake_news_available[agent_state.user_id] = fake_news_ready
            
            if fake_news_ready:
                print(f"✅ Fake news files ready for user {agent_state.user_id}")
                # Set in state machine if available
                if hasattr(agent_state, 'state_machine') and agent_state.state_machine:
                    agent_state.state_machine.set_fake_news_stimulus_available(agent_state.user_id)
            else:
                print(f"❌ Fake news files not ready for user {agent_state.user_id}")
                
            print(f"File availability check complete for user {agent_state.user_id}")
            
        except Exception as e:
            print(f"Error in async file processing: {e}")
            # Cache negative result on error
            self._fake_news_available[agent_state.user_id] = False
    
    async def process_missing_files_sequentially(self, user_id: str, jpg_missing: bool, mp4_missing: bool):
        """
        Process missing files sequentially to avoid facefusion conflicts
        """
        try:
            if jpg_missing and mp4_missing:
                print(f"Both files missing for user {user_id} - processing sequentially...")
                
                # Step 1: Generate JPG first
                print(f"Step 1: Generating JPG for user {user_id}")
                await self.trigger_faceswap_async(user_id)
                
                # Step 2: Wait 10 seconds
                print(f"Waiting 10 seconds before video processing...")
                await asyncio.sleep(10)
                
                # Step 3: Generate MP4
                print(f"Step 2: Generating MP4 for user {user_id}")
                await self.trigger_faceswap_video_async(user_id)
                
                print(f"Sequential processing complete for user {user_id}")
                
            elif jpg_missing:
                print(f"Only JPG missing for user {user_id}")
                await self.trigger_faceswap_async(user_id)
                
            elif mp4_missing:
                print(f"Only MP4 missing for user {user_id}")
                await self.trigger_faceswap_video_async(user_id)
                
        except Exception as e:
            print(f"Error in sequential file processing: {e}")
    
    async def trigger_faceswap_async(self, user_id: str):
        """Trigger faceswap API when JPG is missing"""
        try:
            faceswap_url = f"{self.file_server_url}/faceswap"
            payload = {
                "user_id": user_id,
                "target_face_path": self.target_face_path
            }
            
            print(f"POST {faceswap_url}")
            print(f"Payload: {payload}")
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(faceswap_url, json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"Faceswap successful for user {user_id}")
                    print(f"Response: {result}")
                else:
                    print(f"Faceswap failed: HTTP {response.status_code}")
                    print(f"Response: {response.text}")
                    
        except httpx.TimeoutException:
            print(f"Faceswap timeout for user {user_id}")
        except httpx.ConnectError:  
            print(f"Faceswap connection error for user {user_id}")
        except Exception as e:
            print(f"Error triggering faceswap: {e}")
    
    async def trigger_faceswap_video_async(self, user_id: str):
        """Trigger faceswap-video API when MP4 is missing"""
        try:
            faceswap_video_url = f"{self.file_server_url}/faceswap-video"
            payload = {
                "user_id": user_id,
                "target_video_path": self.target_video_path
            }
            
            print(f"POST {faceswap_video_url}")
            print(f"Payload: {payload}")
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(faceswap_video_url, json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"Faceswap-video successful for user {user_id}")
                    print(f"Response: {result}")
                else:
                    print(f"Faceswap-video failed: HTTP {response.status_code}")
                    print(f"Response: {response.text}")
                    
        except httpx.TimeoutException:
            print(f"Faceswap-video timeout for user {user_id}")
        except httpx.ConnectError:  
            print(f"Faceswap-video connection error for user {user_id}")
        except Exception as e:
            print(f"Error triggering faceswap-video: {e}")