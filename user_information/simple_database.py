import json
import os
from typing import Dict, Optional, Any
from dataclasses import asdict
from .user_profile import UserProfile

class SimpleUserDatabase:
    def __init__(self, db_file: str = "user_profiles.json"):
        self.db_file = db_file
        self.profiles: Dict[str, UserProfile] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for user_id, profile_data in data.items():
                        self.profiles[user_id] = UserProfile(**profile_data)
            except Exception as e:
                print(f"Load error: {e}")

    def _save(self):
        try:
            data = {user_id: asdict(profile) for user_id, profile in self.profiles.items()}
            with open(self.db_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Save error: {e}")

    def get_profile(self, user_id: str) -> Optional[UserProfile]:
        return self.profiles.get(user_id)

    def update_profile(self, user_id: str, extracted_data: Dict[str, Any]) -> UserProfile:
        existing = self.profiles.get(user_id)
        
        if existing:
            # Merge new data with existing
            profile_dict = asdict(existing)
            for key, value in extracted_data.items():
                if value is not None:
                    if key in ['preferred_social_media', 'news_sources']:
                        # Merge lists
                        existing_list = profile_dict.get(key, [])
                        if isinstance(value, list):
                            profile_dict[key] = list(set(existing_list + value))
                    else:
                        profile_dict[key] = value
            updated_profile = UserProfile(**profile_dict)
        else:
            # Create new profile
            updated_profile = UserProfile(user_id=user_id, **extracted_data)
        
        self.profiles[user_id] = updated_profile
        self._save()
        return updated_profile