import json
import os

from config import config

here = os.path.dirname(os.path.abspath(__file__))

class PromptLoader:

    def __init__(self):
        language = config.get('application', 'language')
        conversational_agent_type = config.get('conversational_agent', 'type')
        self.prompts = {}

        prompt_file_name = config.get('prompts','prompts_file') 
        prompt_file = os.path.join(here, prompt_file_name)
        with open(prompt_file, 'r', encoding="utf-8") as file:
            all_prompts = json.load(file)
            self.prompts = all_prompts[language][conversational_agent_type]

    def get_all_prompts(self):
        return self.prompts
    
    def get_prompt(self, prompt_name:str):
        return self.prompts[prompt_name]
    
prompt_loader = PromptLoader()