from conversational_agents.pre_processing.pre_processors.base_pre_processors import BasePreProcessor
from data_models.data_models import AgentState
import requests

class UserProfilePreProcessor(BasePreProcessor):

    def invoke(self, agent_state: AgentState) -> AgentState:
        # print("=== USER PROFILE PRE-PROCESSING CALLED ===")
        # print(f"🔍 DEBUG Pre-Processor:")
        # print(f"   - Input agent_state type: {type(agent_state)}")
        print(f"   - Input agent_state user_id: {agent_state.user_id}")
        
        # Fetch user profile data using the user_id from agent_state
        user_profile_data = self.get_user_profile(agent_state.user_id)
        
        # Store user profile in agent_state for decision agent
        if user_profile_data:
            agent_state.user_profile = user_profile_data
            # print(f"✅ SUCCESS: Set agent_state.user_profile")
            # print(f"   - user_profile type: {type(user_profile_data)}")
            # print(f"   - user_profile keys: {list(user_profile_data.keys())}")
            # print(f"   - agent_state.user_profile: {agent_state.user_profile}")
        else:
            agent_state.user_profile = None
            print(f"❌ FAILED: No user profile data found")
        
        # print(f"🔍 DEBUG Pre-Processor Output:")
        # print(f"   - Output agent_state type: {type(agent_state)}")
        # print(f"   - Output agent_state has user_profile: {hasattr(agent_state, 'user_profile')}")
        # print(f"   - Returning agent_state id: {id(agent_state)}")
        
        return agent_state
    
    def get_user_profile(self, user_id: str):
        """Fetch user profile from user_profile_builder service"""
        try:
            url = f"http://localhost:8010/users/{user_id}"
            # print(f"DEBUG (PreProcessor): Fetching user profile from: {url}")
           
            response = requests.get(url, timeout=5)
            print(f"DEBUG (PreProcessor): Response status code: {response.status_code}")
            
            response.raise_for_status()
            profile_data = response.json()
            
            # Check if the response is the direct user data (has 'user_id' field)
            if 'user_id' in profile_data:
                # print(f"DEBUG (PreProcessor): Found direct user data for {profile_data.get('user_id')}")
                return self.extract_profile_info(profile_data)
            # Check if the response is nested (like {"user456": {...}})
            elif user_id in profile_data:
                print(f"DEBUG (PreProcessor): Found nested user data for {user_id}")
                return self.extract_profile_info(profile_data[user_id])
            else:
                print(f"DEBUG (PreProcessor): User data not found")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"ERROR (PreProcessor): getting user profile: {e}")
            return None
        except Exception as e:
            print(f"ERROR (PreProcessor): processing user profile: {e}")
            return None
    
    def extract_profile_info(self, user_profile):
        """Extract relevant info from user profile"""
        if not user_profile:
            return None
        
        try:
            demographics = user_profile.get('demographics', {})
            fake_news_literacy = user_profile.get('fake_news_literacy', {})
            articulation = user_profile.get('articulation_profile', {})
            personality = user_profile.get('personality_indicators', {})
            emotional_state = user_profile.get('emotional_state', {})
            
            # Extract and clean data
            extracted = {
                'age': demographics.get('age') if demographics.get('age') not in [None, 'unknown'] else None,
                'gender': demographics.get('gender') if demographics.get('gender') not in [None, 'unknown'] else None,
                'school_type': demographics.get('school_type') if demographics.get('school_type') not in [None, 'unknown'] else None,
                'region': demographics.get('region') if demographics.get('region') not in [None, 'unknown'] else None,
                'social_media_usage': demographics.get('social_media_usage') if demographics.get('social_media_usage') not in [None, 'unknown'] else None,
                'interests': demographics.get('interests', []) if demographics.get('interests') else [],
                
                'fake_news_skill': fake_news_literacy.get('self_assessed_skill') if fake_news_literacy.get('self_assessed_skill') not in [None, 'unknown'] else None,
                'fact_checking_habits': fake_news_literacy.get('fact_checking_habits') if fake_news_literacy.get('fact_checking_habits') not in [None, 'unknown'] else None,
                'can_explain_fake_news': fake_news_literacy.get('can_explain_fake_news', False),
                'prior_exposure': fake_news_literacy.get('prior_exposure', []) if fake_news_literacy.get('prior_exposure') else [],
                
                'vocabulary_level': articulation.get('vocabulary_level') if articulation.get('vocabulary_level') not in [None, 'unknown'] else None,
                'expression_style': articulation.get('expression_style') if articulation.get('expression_style') not in [None, 'unknown'] else None,
                'swearing_frequency': articulation.get('swearing_frequency') if articulation.get('swearing_frequency') not in [None, 'unknown'] else None,
                
                'interaction_style': personality.get('interaction_style') if personality.get('interaction_style') not in [None, 'unknown'] else None,
                'attention_span': personality.get('attention_span') if personality.get('attention_span') not in [None, 'unknown'] else None,
                'curiosity_level': personality.get('curiosity_level') if personality.get('curiosity_level') not in [None, 'unknown'] else None,
                
                'current_mood': emotional_state.get('current_mood') if emotional_state.get('current_mood') not in [None, 'unknown', 'neutral'] else None,
                'frustration_level': emotional_state.get('frustration_level') if emotional_state.get('frustration_level') not in [None, 0, 0.0] else None,
                'enthusiasm_level': emotional_state.get('enthusiasm_level') if emotional_state.get('enthusiasm_level') not in [None, 0, 0.0] else None,
            }
            
            # Remove None values to keep data clean
            cleaned = {k: v for k, v in extracted.items() if v is not None and v != [] and v != ''}
            
            print(f"DEBUG (PreProcessor): Extracted profile: {cleaned}")
            return cleaned
            
        except Exception as e:
            print(f"ERROR (PreProcessor): extracting profile info: {e}")
            return None