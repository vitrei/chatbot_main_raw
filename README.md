# Chatbot

This is the repository for the chatbot backend component.

## Installation

### Requirements
- Python 3.11.X

Install for example with conda:

``conda create -n llm_chatbot python=3.11``

Then activate the envorionment

``conda activate llm_chatbot``

### Dependencies
``pip install -r requirements.txt``

### Install NLTK punkt
In terminal run ``python``

then

```bash
import nltk
nltk.download('punkt_tab')
```

then

```bash
exit()
```

### Add config files
Create a file ``config.ini`` from the ``config_example.ini`` file.

Create another file ``dependencies.yaml``. Leave this empty, if you have no custom decision agent or custom actions.

## Run the Chatbot
In the console:

``python run_api.py``

## Access the API
Request
```javascript
POST http://localhost:5000/instruct
{
    "userId","some user id for unique interaction",
    "content": "Here is your instruction to the chatbot",
    "stream", false
}
```

Response
```javascript
{
    "content": "The answer of the chatbot",
    "payload": {
        "any": "payload"
    }
}
```

## Concepts

### Data Models
The data models define the most generic models used. These include the LLM answer, decision types, the decision itself and the agent state.

#### LLMAnswer
The LLM answer cosists of the generated answer text (content) and any custom payload as a dictionary:

```javascript
{
    "content": "Here is the generated LLM answer",
    "payload": {
        "any": "payload",
        "you": {
            "want": [1,2,3,4]
        }
    }
}
```

### Prompts
Prompts are collected in one single file. The path to the file is set in the config.ini. The prompts can separated by languge code (de, en, etc.). Prompts can be specified for simple and RAG conversational agents.

For simple and RAG conversational agents system prompt and guiding instructions can be definied. Guiding instructions are explained below.

For RAG conversational agens a retriever prompt has to be specified, which is sent to the retrievers (retriever_prompt) and a prompt that indicates RAG information in the prompt (rag_prompt).

### Decision Agent
The decision agent is a core component of the conversational agent. Based on the agent state, it decides the next action to do. The default decision agent is the ConversationOnlyDecisionAgent and is loaded when dependencies.yaml does not contain another class for BaseDecisionAgent.

If you want to implement your own decision agent, create a new class that inherits from BaseDecisionAgent and add the path to the new class to the dependencies.yaml file, e.g., BaseDecisionAgent: "conversational_agents.agent_logic.opra_logic.opra_decision_agent.OPRADecisionAgent"

The decision agent has always access to the conversational agent state and next actions decision types are defined in the data model NextActionDecisionType and can be from the categories:

 - GENERATE_ANSWER: generate an answer with the LLM from the instruction
 - PROMPT_ADAPTION: adapt the (system) prompt in some way an then generate an answer with the LLM
 - GUIDING_INSTRUCTIONS: add a specified guiding instruction to the instruction and generate an answer
 - ACTION: call an external action, e.g., API call, another LLM etc. Custom logic can be implemented this way

The decision agent can look like this:

```python
from data_models.data_models import AgentState, NextActionDecision, NextActionDecisionType
from conversational_agents.agent_logic.base_decision_agent import BaseDecisionAgent

class OPRADecisionAgent(BaseDecisionAgent):

    def next_action(self, agent_state: AgentState):

        next_action_decision = NextActionDecision(
            type=NextActionDecisionType.GENERATE_ANSWER,
            action=None,
            payload=None
        )        

        if agent_state.conversation_turn_counter == 3:
            next_action_decision.type = NextActionDecisionType.GUIDING_INSTRUCTIONS
            next_action_decision.action = "location"
        
        elif agent_state.conversation_turn_counter == 4:
            next_action_decision.type = NextActionDecisionType.ACTION
            next_action_decision.action = "path_prediction"

        elif agent_state.conversation_turn_counter == 5:
            next_action_decision.type = NextActionDecisionType.ACTION
            next_action_decision.action = "parrot"

        else:
            next_action_decision.type = NextActionDecisionType.GUIDING_INSTRUCTIONS
            next_action_decision.action = "general_guidance"
        
        return next_action_decision 
```

### Actions
It is possibe to call, based on the decision of the decision agent, external actions, for instance APIs.
All callable actions has to be defined in a class that inherits from BaseConversationalAgentActionsCollection. This new class has to be listed in the dependencies.yaml file, e.g. BaseConversationalAgentActionsCollection: "conversational_agents.agent_logic.opra_logic.opra_action_collection.ConversationalAgentActionCollection"

Each action must have a method invoke(state: AgentState) -> LLMAnswer. The following example is an action that returns an LLM answer.

```python
from data_models.data_models import AgentState, LLMAnswer


class PathPredictionAction:

    def __init__(self):
        pass

    def invoke(self, agent_state:AgentState) -> LLMAnswer:
        llm_answer = LLMAnswer(
            content="Du musst Informatik an der HKA studieren!",
            payload={
                "type":"educationalPath", 
                "data": {
                    "title": "Informatik",
                    "description": "Lorem Ipsum"
                }
            }
        )

        return llm_answer
```

Actions like that can then be listed in the action collection class like this:

```python
from conversational_agents.agent_logic.base_conversational_agent_action_collection import BaseConversationalAgentActionsCollection
from conversational_agents.agent_logic.opra_logic.opra_actions.parrot_action import ParrotAction
from conversational_agents.agent_logic.opra_logic.opra_actions.path_recommendation_action import PathPredictionAction

class ConversationalAgentActionCollection(BaseConversationalAgentActionsCollection):

    def __init__(self):
        super().__init__()

        self.actions = {
            "path_prediction": PathPredictionAction(),
            "parrot": ParrotAction()
        }

    def get_action(self, action_name):
        if action_name not in self.actions:
            return None
        return self.actions[action_name]
    
    def get_actions(self):
        return self.actions
```