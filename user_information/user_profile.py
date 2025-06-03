from dataclasses import dataclass
from typing import List, Optional, Dict, Any

@dataclass
class UserProfile:
    user_id: str
    age: Optional[int] = None
    gender: Optional[str] = None
    location: Optional[str] = None
    school_type: Optional[str] = None
    grade: Optional[str] = None  
    
    preferred_social_media: List[str] = None
    daily_internet_hours: Optional[str] = None
    news_sources: List[str] = None
    
    fact_checking_awareness: Optional[str] = None  
    source_verification_habits: Optional[str] = None
    critical_thinking_level: Optional[str] = None
    
    def __post_init__(self):
        if self.preferred_social_media is None:
            self.preferred_social_media = []
        if self.news_sources is None:
            self.news_sources = []