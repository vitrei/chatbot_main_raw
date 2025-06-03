from conversational_agents.conversational_agents_handler import ConversationalAgentsHandler
from user_information import UserInformationExtractionService, BackgroundUserInfoProcessor

# Globale Services
_user_info_service = None
_background_processor = None

def initialize_user_info_extraction():
    """Initialisiert User Information Extraction Services"""
    global _user_info_service, _background_processor
    
    if _user_info_service is None:
        _user_info_service = UserInformationExtractionService()
        _background_processor = BackgroundUserInfoProcessor(_user_info_service)
        _background_processor.start()
        print("User Information Extraction Service started")

def extract_user_info_after_turn(user_id: str, chat_history):
    """Extrahiert User Information nach einem Conversation Turn"""
    if _background_processor:
        _background_processor.queue_extraction(user_id, chat_history)

# Monkey Patch: Erweitere die bestehende get_by_user_id Methode
original_get_by_user_id = ConversationalAgentsHandler.get_by_user_id

def enhanced_get_by_user_id(self, user_id: str, decision_agent):
    """Erweiterte get_by_user_id mit User Information Extraction"""
    
    # Hole den ursprünglichen Agent
    conversational_agent = original_get_by_user_id(self, user_id, decision_agent)
    
    # Erweitere instruct Methode
    original_instruct = conversational_agent.instruct
    
    async def enhanced_instruct(instruction: str):
        result = await original_instruct(instruction)
        # Extrahiere User Info nach dem Turn
        extract_user_info_after_turn(user_id, conversational_agent.state.chat_history)
        return result
    
    # Erweitere stream Methode  
    original_stream = conversational_agent.stream
    
    async def enhanced_stream(instruction: str):
        async for chunk in original_stream(instruction):
            yield chunk
        # Extrahiere User Info nach dem Stream
        extract_user_info_after_turn(user_id, conversational_agent.state.chat_history)
    
    # Überschreibe die Methoden
    conversational_agent.instruct = enhanced_instruct
    conversational_agent.stream = enhanced_stream
    
    return conversational_agent

# Wende Monkey Patch an
ConversationalAgentsHandler.get_by_user_id = enhanced_get_by_user_id