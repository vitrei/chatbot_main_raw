import yaml
import importlib
from conversational_agents.agent_logic.base_conversational_agent_action_collection import BaseConversationalAgentActionsCollection
from conversational_agents.agent_logic.base_decision_agent import BaseDecisionAgent
from conversational_agents.conversational_agents_handler import ConversationalAgentsHandler
from conversational_agents.post_processing.post_processing_pipeline import PostProcessingPipeline
from conversational_agents.pre_processing.pre_processing_pipeline import PreProcessingPipeline



def dynamic_import(class_path: str):
    """Dynamically import a class or variable from a module string path."""
    module_name, class_name = class_path.rsplit(".", 1) 
    module = importlib.import_module(module_name)  
    return getattr(module, class_name) 

with open("dependencies.yaml", "r") as file:
    config = yaml.safe_load(file)

actions = {}
if config != None and "actions" in config and config["actions"] != None and config["actions"] != "":
    actions = dynamic_import(config["actions"])

class ConversationalAgentsHandlerFactory():

    def __init__(self):
        pass

    def create(self):

        if config != None and 'BaseConversationalAgentActionsCollection' in config:
            ConversationalAgentsActionsClass = dynamic_import(config["BaseConversationalAgentActionsCollection"])
            if not issubclass(ConversationalAgentsActionsClass, BaseConversationalAgentActionsCollection):
                raise TypeError(f"{ConversationalAgentsActionsClass.__name__} must be a subclass of BaseConversationalAgentActionsCollection")

            conversational_agent_actions = ConversationalAgentsActionsClass()
        else:
            conversational_agent_actions = None

        post_processor_paths = config.get("PostProcessors", [])
        if post_processor_paths == None:
            post_processor_paths = []
        list_of_post_processors = [dynamic_import(path)() for path in post_processor_paths]

        pre_processor_paths = config.get("PreProcessors", [])
        if pre_processor_paths == None:
            pre_processor_paths = []
        list_of_pre_processors = [dynamic_import(path)() for path in pre_processor_paths]
        

        print(list_of_pre_processors)
        print(list_of_post_processors)

        post_processing_pipeline = PostProcessingPipeline(list_of_post_processors)

        pre_processing_pipeline = PreProcessingPipeline(list_of_pre_processors)

        conversational_agents_hander = ConversationalAgentsHandler(agent_logic_actions=conversational_agent_actions, post_processing_pipeline=post_processing_pipeline, pre_processing_pipeline=pre_processing_pipeline)

        return conversational_agents_hander

class DecisionAgentFactory():
     
     def __init__(self):
          pass
     
     def create(self):
        
        if config == None or 'BaseDecisionAgent' not in config:
            DecisionAgentClass = dynamic_import('conversational_agents.agent_logic.general_logic.conversation_only_decision_agent.ConversationOnlyDecisionAgent')
        else:
            DecisionAgentClass = dynamic_import(config["BaseDecisionAgent"])
        if not issubclass(DecisionAgentClass, BaseDecisionAgent):
            raise TypeError(f"{DecisionAgentClass.__name__} must be a subclass of BaseDecisionAgent")

        decision_agent_instance = DecisionAgentClass()

        return decision_agent_instance

