import requests
from typing import Optional, Dict, Any
from conversational_agents.pre_processing.pre_processors.base_pre_processors import BasePreProcessor
from data_models.data_models import AgentState

class UserProfilePreProcessor(BasePreProcessor):
    
    def __init__(self, timeout: float = 3.0, max_retries: int = 2):
        """
        Initialize with configurable timeout and retry settings
        
        Args:
            timeout: Maximum time to wait for user profile (seconds)
            max_retries: Number of retry attempts on failure
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_profile_service_url = "http://localhost:8010"
        
    def invoke(self, agent_state: AgentState) -> AgentState:
        """
        Invoke pre-processing with robust error handling
        Always returns agent_state (with or without user_profile)
        """
        print("=== USER PROFILE PRE-PROCESSING ===")
        print(f"Processing user_id: {agent_state.user_id}")
        print(f"Timeout: {self.timeout}s, Max retries: {self.max_retries}")
        
        # Try to get user profile with retries
        user_profile_data = self.get_user_profile_with_retries(agent_state.user_id)
            
        # Always set user_profile (None if failed)
        if user_profile_data:
            agent_state.user_profile = user_profile_data
            # print(f"SUCCESS: User profile loaded with {len(user_profile_data)} fields")
            # print(f"Profile keys: {list(user_profile_data.keys())}")
        else:
            agent_state.user_profile = None
            print(f"WARNING: No user profile available - continuing with default behavior")
        
        print("=== PRE-PROCESSING COMPLETE ===")
        return agent_state
    
    def get_user_profile_with_retries(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch user profile with robust error handling and retries
        
        Returns:
            Dict with user profile data or None if failed
        """
        
        for attempt in range(self.max_retries + 1):
            try:
                print(f"Attempt {attempt + 1}/{self.max_retries + 1}: Fetching user profile...")
                
                url = f"{self.user_profile_service_url}/users/{user_id}"
                print(f"GET {url}")
                
                response = requests.get(url, timeout=self.timeout)
                print(f"Response: {response.status_code}")
                
                if response.status_code == 200:
                    profile_data = response.json()
                    processed_profile = self.extract_profile_info(profile_data, user_id)
                    
                    if processed_profile:
                        print(f"Success on attempt {attempt + 1}")
                        return processed_profile
                    else:
                        print(f"Empty profile data on attempt {attempt + 1}")
                        
                elif response.status_code == 404:
                    print(f"User {user_id} not found - no retries needed")
                    return None  # Don't retry for 404
                    
                else:
                    print(f"HTTP {response.status_code} on attempt {attempt + 1}")
                    
            except requests.exceptions.Timeout:
                print(f"TIMEOUT on attempt {attempt + 1} (>{self.timeout}s)")
                
            except requests.exceptions.ConnectionError:
                print(f"CONNECTION ERROR on attempt {attempt + 1}")
                
            except Exception as e:
                print(f"UNEXPECTED ERROR on attempt {attempt + 1}: {e}")
            
            if attempt < self.max_retries:
                import time
                wait_time = 0.5 * (2 ** attempt)  # 0.5s, 1s, 2s...
                print(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
        
        print(f"All {self.max_retries + 1} attempts failed")
        return None
    
    def extract_profile_info(self, raw_data: Dict[str, Any], user_id: str) -> Optional[Dict[str, Any]]:
        """
        Extract and clean profile information from raw API response
        
        Args:
            raw_data: Raw response from user profile service
            user_id: User ID for logging
            
        Returns:
            Cleaned profile dict or None if extraction failed
        """
        try:
            user_data = None
            
            if 'user_id' in raw_data:
                print(f"Found direct user data format")
                user_data = raw_data
                
            elif user_id in raw_data:
                print(f"Found nested user data format")
                user_data = raw_data[user_id]
                
            # Try to find user data in 'data' field
            elif 'data' in raw_data and user_id in raw_data['data']:
                print(f"Found user data in 'data' field")
                user_data = raw_data['data'][user_id]
                
            else:
                print(f"No user data found in response format")
                print(f"Available keys: {list(raw_data.keys())}")
                return None
            
            if not user_data:
                print(f"User data is empty")
                return None
            
            demographics = user_data.get('demographics', {})
            fake_news_literacy = user_data.get('fake_news_literacy', {})
            articulation = user_data.get('articulation_profile', {})
            personality = user_data.get('personality_indicators', {})
            emotional_state = user_data.get('emotional_state', {})
            
            extracted = {
                'age': self.safe_get(demographics, 'age'),
                'gender': self.safe_get(demographics, 'gender'),
                'school_type': self.safe_get(demographics, 'school_type'),
                'region': self.safe_get(demographics, 'region'),
                'social_media_usage': self.safe_get(demographics, 'social_media_usage'),
                'interests': demographics.get('interests', []) if demographics.get('interests') else [],
                
                'fake_news_skill': self.safe_get(fake_news_literacy, 'self_assessed_skill'),
                'fact_checking_habits': self.safe_get(fake_news_literacy, 'fact_checking_habits'),
                'can_explain_fake_news': fake_news_literacy.get('can_explain_fake_news', False),
                'prior_exposure': fake_news_literacy.get('prior_exposure', []) if fake_news_literacy.get('prior_exposure') else [],
                
                'vocabulary_level': self.safe_get(articulation, 'vocabulary_level'),
                'expression_style': self.safe_get(articulation, 'expression_style'),
                'swearing_frequency': self.safe_get(articulation, 'swearing_frequency'),
                
                'interaction_style': self.safe_get(personality, 'interaction_style'),
                'attention_span': self.safe_get(personality, 'attention_span'),
                'curiosity_level': self.safe_get(personality, 'curiosity_level'),
                
                'current_mood': self.safe_get(emotional_state, 'current_mood'),
                'frustration_level': emotional_state.get('frustration_level') if emotional_state.get('frustration_level') not in [None, 0, 0.0] else None,
                'enthusiasm_level': emotional_state.get('enthusiasm_level') if emotional_state.get('enthusiasm_level') not in [None, 0, 0.0] else None,
            }
            
            cleaned = {k: v for k, v in extracted.items() if v is not None and v != '' and v != []}
            
            if cleaned:
                print(f"Extracted {len(cleaned)} profile fields: {list(cleaned.keys())}")
                return cleaned
            else:
                print(f"No meaningful profile data extracted")
                return None
                
        except Exception as e:
            print(f"Error extracting profile info: {e}")
            return None
    
    def safe_get(self, data: Dict[str, Any], key: str) -> Optional[Any]:
        """
        Safely get value from dict, filtering out 'unknown', None, empty strings
        
        Args:
            data: Dictionary to extract from
            key: Key to extract
            
        Returns:
            Value or None if not meaningful
        """
        value = data.get(key)
        
        if value in [None, 'unknown', '', 'null', 'undefined']:
            return None
            
        return value