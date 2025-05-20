import json
import random
from langchain_core.language_models import BaseChatModel
from langchain_ollama.chat_models import ChatOllama
from langchain_openai.chat_models.base import ChatOpenAI

from config import config

class LLMFactory():

    def __init__(self):
        self.model_name = config.get("llm", "model_name")
        print('model_name:', self.model_name)

    def get_llm(self, model_name=None):

        current_model_name = self.model_name
        if model_name != None:
            current_model_name = model_name            

        llm = None

        if current_model_name in ['gemma3:12b']:

            urls = json.loads(config.get("llm","host_names_hka"))
            chat_llm_url = random.choice(urls)
            
            llm = ChatOllama(
                model = self.model_name,
                base_url = chat_llm_url,
                keep_alive = -1 # any negative number which will keep the model loaded in memory (e.g. -1 or "-1m")
            )

        elif current_model_name in ['llama3.1:8b_ionos']:

            urls = json.loads(config.get("llm","host_names_ionos_model_hub"))
            chat_llm_url = random.choice(urls)

            llm = ChatOpenAI(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                openai_api_base=chat_llm_url,
                openai_api_key=config.get("llm", "ionos_model_hub_api_key")
            )

        elif current_model_name in ['llama3']:
            urls = json.loads(config.get("llm","host_names_ionos"))
            chat_llm_url = random.choice(urls)
            
            llm = ChatOllama(
                model = self.model_name,
                base_url = chat_llm_url,
                keep_alive = -1 # any negative number which will keep the model loaded in memory (e.g. -1 or "-1m")
            )
            
        elif current_model_name in ['openGPT-X/Teuken-7B-instruct-commercial-v0.4']:

            opengptx_urls = json.loads(config.get("llm","host_names_opengptx"))
            chat_llm_url = random.choice(opengptx_urls)
            vllm_api_key = config.get("llm", "vllm_api_key")
            
            print("server:", chat_llm_url)
            print("model:", self.model_name)
            print("api_key", vllm_api_key)

            llm = ChatOpenAI(
                model=self.model_name,
                openai_api_key=vllm_api_key,
                openai_api_base=chat_llm_url,
                max_tokens=250
            )

        return llm

llm_factory = LLMFactory()