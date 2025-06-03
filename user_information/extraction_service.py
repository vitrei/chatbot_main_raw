from typing import Dict, Any
from langchain.schema import HumanMessage, AIMessage
from langchain_core.messages.ai import AIMessageChunk
from .information_extractor import UserInformationExtractor
from .simple_database import SimpleUserDatabase
from .user_profile import UserProfile

class UserInformationExtractionService:
    """
    Einfacher Service zur Extraktion von Benutzerinformationen
    Läuft unabhängig von der Chatbot-Logik
    """
    
    def __init__(self, db_file: str = "user_profiles.json"):
        self.extractor = UserInformationExtractor()
        self.database = SimpleUserDatabase(db_file)

    def process_conversation_turn(self, user_id: str, chat_history: Dict[str, Any]) -> UserProfile:
        """
        Verarbeitet die Chat-Historie nach jedem Conversation Turn
        
        Args:
            user_id: Eindeutige Benutzer-ID
            chat_history: Complete chat history from AgentState
            
        Returns:
            Updated UserProfile
        """
        # Convert chat history to readable text
        conversation_text = self._format_chat_history(chat_history)
        
        # Extract user information
        extracted_data = self.extractor.extract_info(conversation_text)
        
        # Update profile in database
        updated_profile = self.database.update_profile(user_id, extracted_data)
        
        print(f"User info extracted for {user_id}: {extracted_data}")
        
        return updated_profile

    def get_user_profile(self, user_id: str) -> UserProfile:
        """Get current user profile"""
        return self.database.get_profile(user_id)

    def _format_chat_history(self, chat_history: Dict[str, Any]) -> str:
        """Convert chat history to text format"""
        conversation = ""
        
        for session_id, history in chat_history.items():
            if hasattr(history, 'messages'):
                for message in history.messages:
                    if isinstance(message, HumanMessage):
                        conversation += f"User: {message.content}\n"
                    elif isinstance(message, (AIMessage, AIMessageChunk)):
                        conversation += f"Bot: {message.content}\n"
        
        return conversation.strip()