import requests
from typing import Dict, Any, Optional
from conversational_agents.pre_processing.pre_processors.base_pre_processors import BasePreProcessor
from data_models.data_models import AgentState

class FakeNewsPreProcessor(BasePreProcessor):
    def __init__(self, file_server_url: str = "http://localhost:8000"):
        self.file_server_url = file_server_url
        self._fetched_content = {}
        print(f"FakeNewsPreProcessor initialized with server: {file_server_url}")
    
    def check_fake_news_availability(self, user_id: str) -> Dict[str, Any]:
        """Check if fake news files are available for the user"""
        try:
            response = requests.get(f"{self.file_server_url}/check-file/{user_id}")
            if response.status_code == 200:
                return response.json()
            else:
                print(f"File check failed with status {response.status_code}")
                return {"jpg_exists": False, "mp4_exists": False}
        except Exception as e:
            print(f"Error checking fake news files: {e}")
            return {"jpg_exists": False, "mp4_exists": False}
    
    def fetch_fake_news_file(self, user_id: str, file_type: str = "mp4") -> Optional[str]:
        """Fetch fake news file and return the file path or URL"""
        try:
            url = f"{self.file_server_url}/get-file-info/{user_id}/{file_type}"  # Use the new endpoint
            print(f"DEBUG: Making API call to: {url}")
            
            response = requests.get(url)
            print(f"DEBUG: Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"DEBUG: File info received: {result}")
                return result.get("file_path") or result.get("file_url")
            else:
                print(f"File info fetch failed with status {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error fetching fake news file info: {e}")
            return None
    
    def get_fake_news_data(self, user_id: str) -> Dict[str, Any]:
        """Get fake news data for the user with caching logic"""
        try:
            # Check if we already have fetched content for this user
            if user_id in self._fetched_content:
                cached_content = self._fetched_content[user_id]
                print(f"DEBUG: Using cached fake news data for user {user_id}: {cached_content['type']} file")
                return cached_content
            
            # Check availability on server
            fake_news_check = self.check_fake_news_availability(user_id)
            
            # If no content available, return early without caching
            if not fake_news_check.get("mp4_exists", False) and not fake_news_check.get("jpg_exists", False):
                print(f"DEBUG: No fake news files available for user {user_id}")
                return {"available": False}
            
            # Content is available, fetch it
            fake_news_data = None
            if fake_news_check.get("mp4_exists", False):
                print(f"DEBUG: Fetching fake news MP4 for user {user_id}")
                fake_news_path = self.fetch_fake_news_file(user_id, "mp4")
                if fake_news_path:
                    fake_news_data = {
                        "type": "mp4",
                        "path": fake_news_path,
                        "available": True,
                        "fetched_at": user_id  # Simple tracking
                    }
            elif fake_news_check.get("jpg_exists", False):
                print(f"DEBUG: Fetching fake news JPG for user {user_id}")
                fake_news_path = self.fetch_fake_news_file(user_id, "jpg")
                if fake_news_path:
                    fake_news_data = {
                        "type": "jpg", 
                        "path": fake_news_path,
                        "available": True,
                        "fetched_at": user_id  # Simple tracking
                    }
            
            # Cache the fetched content
            if fake_news_data:
                self._fetched_content[user_id] = fake_news_data
                print(f"DEBUG: Cached fake news data for user {user_id}: {fake_news_data['type']} file")
                return fake_news_data
            
            return {"available": False}
            
        except Exception as e:
            print(f"ERROR: Could not get fake news data for user {user_id}: {e}")
            return {"available": False, "error": str(e)}
    
    def invoke(self, agent_state: AgentState) -> AgentState:
        """Main preprocessing method - fetch fake news data and add to agent state"""
        try:
            # Get user ID from agent state or use fallback
            user_id = getattr(agent_state, 'user_id', 'test_2001')
            
            print(f"DEBUG: FakeNewsPreProcessor processing user {user_id}")
            
            # Get fake news data (with caching logic)
            fake_news_data = self.get_fake_news_data(user_id)
            
            # Only set agent state if there's available content
            if fake_news_data.get("available"):
                agent_state.fake_news_data = fake_news_data
                print(f"DEBUG: Set fake news data in agent state: {fake_news_data['type']} file")
            else:
                # Don't set anything if no content is available
                print(f"DEBUG: No fake news content to set for user {user_id}")
                # Optionally, you can explicitly set it to None or leave it unset
                if not hasattr(agent_state, 'fake_news_data'):
                    agent_state.fake_news_data = None
            
            return agent_state
            
        except Exception as e:
            print(f"ERROR in FakeNewsPreProcessor: {e}")
            # Don't set fake_news_data on error
            return agent_state
    
    def clear_cache_for_user(self, user_id: str):
        """Clear cached content for a specific user (call when new content is available)"""
        if user_id in self._fetched_content:
            del self._fetched_content[user_id]
            print(f"DEBUG: Cleared fake news cache for user {user_id}")
    
    def clear_all_cache(self):
        """Clear all cached content"""
        self._fetched_content.clear()
        print("DEBUG: Cleared all fake news cache")