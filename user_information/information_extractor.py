import json
import re
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from large_language_models.llm_factory import llm_factory

class UserInformationExtractor:
    def __init__(self):
        self.extraction_prompt = """
Du analysierst Gespräche mit Jugendlichen zwischen 14-18 Jahren zum Thema Fake News und Medienkompetenz.

Gesprächsverlauf:
{conversation_history}

Extrahiere nur explizit erwähnte Informationen als JSON:

{{
    "age": "Alter als Zahl oder null",
    "gender": "Geschlecht oder null", 
    "location": "Wohnort/Stadt oder null",
    "school_type": "Schultyp (Gymnasium, Realschule, etc.) oder null",
    "grade": "Klassenstufe oder null",
    "preferred_social_media": ["Instagram", "TikTok", "etc."],
    "daily_internet_hours": "Stunden pro Tag oder null",
    "news_sources": ["wo sie Nachrichten lesen/schauen"],
    "fact_checking_awareness": "hoch/mittel/niedrig oder null",
    "source_verification_habits": "beschreibung oder null",
    "critical_thinking_level": "hoch/mittel/niedrig oder null"
}}

Wichtig: NUR explizit erwähnte Fakten extrahieren, keine Vermutungen!
"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Du extrahierst Informationen über Jugendliche aus Gesprächen über Medienkompetenz."),
            ("human", self.extraction_prompt)
        ])
        
        llm = llm_factory.get_llm()
        self.chain = prompt | llm

    def extract_info(self, conversation_text: str) -> Dict[str, Any]:
        try:
            response = self.chain.invoke({
                "conversation_history": conversation_text
            })
            
            json_str = self._extract_json(response.content)
            if json_str:
                return json.loads(json_str)
        except Exception as e:
            print(f"Extraction error: {e}")
        return {}

    def _extract_json(self, text: str) -> str:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        return match.group(0) if match else None