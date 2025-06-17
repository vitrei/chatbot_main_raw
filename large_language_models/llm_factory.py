import json
import random
from functools import lru_cache
from langchain_core.language_models import BaseChatModel
from langchain_ollama.chat_models import ChatOllama
from langchain_openai.chat_models.base import ChatOpenAI
import os

from config import config

class LLMFactory():
    _instance = None
    _llm_instances = {}
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if self._instance is not None:
            raise RuntimeError("Use get_instance() instead")
        self.model_name = config.get("llm", "model_name")
        print(f'LLMFactory initialized with model: {self.model_name}')
        # Pre-create default LLM
        self._create_llm(self.model_name)

    def get_llm(self, model_name=None):
        current_model_name = model_name or self.model_name
        
        if current_model_name not in self._llm_instances:
            self._create_llm(current_model_name)
            
        return self._llm_instances.get(current_model_name)
    
    def _create_llm(self, model_name):
        """Internal method to create and cache an LLM instance"""
        try:
            llm = None
            
            if model_name == 'openai':
                openai_model = config.get("llm", "openai_model", fallback="gpt-4o")
                api_key = config.get("llm", "openai_api_key")
                
                if not api_key or api_key == "your_openai_api_key_here":
                    api_key = os.environ.get("OPENAI_API_KEY")
                    if not api_key:
                        raise ValueError("No valid OpenAI API key found")
                
                llm = ChatOpenAI(
                    model=openai_model,
                    openai_api_key=api_key,
                    temperature=0.7
                )
                
            elif model_name in ['gemma3:27b']:
                urls = json.loads(config.get("llm", "host_names_hka"))
                chat_llm_url = random.choice(urls)
                
                llm = ChatOllama(
                    model=model_name,
                    base_url=chat_llm_url,
                    keep_alive=-1
                )

            if llm is None:
                raise ValueError(f"Failed to create LLM for model: {model_name}")
            
            self._llm_instances[model_name] = llm
            return llm
                
        except Exception as e:
            print(f"Error creating LLM for {model_name}: {e}")
            # Try fallback to OpenAI if available
            try:
                api_key = os.environ.get("OPENAI_API_KEY")
                if api_key:
                    fallback_llm = ChatOpenAI(
                        model="gpt-4o",
                        openai_api_key=api_key,
                        temperature=0.7
                    )
                    self._llm_instances[model_name] = fallback_llm
                    return fallback_llm
            except:
                pass
            return None

# Create singleton instance
llm_factory = LLMFactory.get_instance()
