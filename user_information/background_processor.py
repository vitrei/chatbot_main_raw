import threading
import time
from typing import Dict, Any
from .extraction_service import UserInformationExtractionService

class BackgroundUserInfoProcessor:
    """
    Background processor that runs user information extraction
    independently from the main chatbot logic
    """
    
    def __init__(self, extraction_service: UserInformationExtractionService):
        self.extraction_service = extraction_service
        self.pending_extractions = {}
        self.running = False
        self.thread = None

    def start(self):
        """Start background processing"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._process_loop, daemon=True)
            self.thread.start()
            print("Background user info processor started")

    def stop(self):
        """Stop background processing"""
        self.running = False
        if self.thread:
            self.thread.join()
        print("Background user info processor stopped")

    def queue_extraction(self, user_id: str, chat_history: Dict[str, Any]):
        """Queue a user information extraction"""
        self.pending_extractions[user_id] = chat_history

    def _process_loop(self):
        """Main processing loop"""
        while self.running:
            if self.pending_extractions:
                user_id, chat_history = self.pending_extractions.popitem()
                try:
                    self.extraction_service.process_conversation_turn(user_id, chat_history)
                except Exception as e:
                    print(f"Background extraction error for {user_id}: {e}")
            
            time.sleep(1)  # Check every second