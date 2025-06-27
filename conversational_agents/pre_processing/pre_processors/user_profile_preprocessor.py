import asyncio
import httpx
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
        Invoke pre-processing with async user profile fetching
        Returns agent_state immediately, profile loads asynchronously
        """
        print("=== USER PROFILE PRE-PROCESSING ===")
        print(f"Processing user_id: {agent_state.user_id}")
        print(f"Timeout: {self.timeout}s, Max retries: {self.max_retries}")
        
        # Start async profile loading (non-blocking)
        asyncio.create_task(self.load_user_profile_async(agent_state))
        
        # Return immediately with empty profile (will be populated async)
        agent_state.user_profile = None
        print("Pre-processing complete - profile loading in background")
        return agent_state

    async def load_user_profile_async(self, agent_state: AgentState):
        """
        Load user profile asynchronously and update agent_state
        """
        try:
            user_profile_data = await self.get_user_profile_with_retries_async(agent_state.user_id)
            
            if user_profile_data:
                agent_state.user_profile = user_profile_data
                print(f"User profile loaded asynchronously for {agent_state.user_id}")
            else:
                agent_state.user_profile = None
                print(f"No user profile available for {agent_state.user_id}")
                
        except Exception as e:
            print(f"Error loading user profile async: {e}")
            agent_state.user_profile = None

    async def get_user_profile_with_retries_async(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch user profile with robust error handling and retries (async version)
        
        Returns:
            Dict with user profile data or None if failed
        """
        
        for attempt in range(self.max_retries + 1):
            try:
                print(f"Attempt {attempt + 1}/{self.max_retries + 1}: Fetching user profile...")
                
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    url = f"{self.user_profile_service_url}/users/{user_id}"
                    print(f"GET {url}")
                    response = await client.get(url)
                    print(f"Response: {response.status_code}")
                    
                    if response.status_code == 200:
                        profile_data = response.json()
                        processed_profile = self.extract_profile_info(profile_data, user_id)
                        if processed_profile:
                            print(f"Success on attempt {attempt + 1}")
                            return processed_profile
                        else:
                            print(f"Empty profile data on attempt {attempt + 1}")
                            
                    elif response.status_code == 404 or response.status_code == 500:
                        print(f"User {user_id} not found (HTTP {response.status_code}) - creating user with demographics...")
                        
                        # Call create-user-with-demographics endpoint
                        try:
                            create_url = f"{self.user_profile_service_url}/create-user-with-demographics/{user_id}"
                            print(f"POST {create_url}")
                            create_response = await client.post(create_url)
                            print(f"Create response: {create_response.status_code}")
                            
                            if create_response.status_code == 200:
                                # User created successfully, extract the profile
                                create_result = create_response.json()
                                raw_profile = create_result.get("profile")
                                
                                if raw_profile:
                                    # Process the profile using your existing method
                                    processed_profile = self.extract_profile_info({"profile": raw_profile}, user_id)
                                    if processed_profile:
                                        print(f"Successfully created user {user_id} with demographics - Age: {create_result.get('profile', {}).get('demographics', {}).get('age', 'unknown')}, Gender: {create_result.get('profile', {}).get('demographics', {}).get('gender', 'unknown')}")
                                        return processed_profile
                                    else:
                                        print(f"Failed to process created profile for user {user_id}")
                                else:
                                    print(f"No profile data in creation response for user {user_id}")
                                    
                            else:
                                print(f"Failed to create user {user_id}: HTTP {create_response.status_code}")
                                if create_response.status_code == 404:
                                    print("No images available for user creation with demographics")
                                elif create_response.status_code == 500:
                                    print("Error during demographics analysis or user creation")
                                
                        except httpx.RequestError as create_error:
                            print(f"Error during user creation with demographics: {create_error}")
                        
                        # Return None after creation attempt (whether successful or not)
                        return None
                        
                    else:
                        print(f"HTTP {response.status_code} on attempt {attempt + 1}")
                        
            except httpx.TimeoutException:
                print(f"TIMEOUT on attempt {attempt + 1} (>{self.timeout}s)")
            except httpx.ConnectError:
                print(f"CONNECTION ERROR on attempt {attempt + 1}")
            except Exception as e:
                print(f"UNEXPECTED ERROR on attempt {attempt + 1}: {e}")
                
            if attempt < self.max_retries:
                wait_time = 0.5 * (2 ** attempt)
                print(f"Waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)
        
        print(f"All {self.max_retries + 1} attempts failed")
        return None

    # Keep your existing extraction methods unchanged
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