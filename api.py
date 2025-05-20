
from typing import Annotated
from fastapi import Depends, FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import json

from dependency_injection import ConversationalAgentsHandlerFactory, DecisionAgentFactory


app = FastAPI()

# dependency injection
conversational_agents_handler_factory = ConversationalAgentsHandlerFactory()
decision_agent_factory = DecisionAgentFactory()
conversational_agents_handler = conversational_agents_handler_factory.create()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def info():
    json_str = json.dumps({"api":"LLM chatbot backen running", "version":"1.0.0"}, default=str)
    return Response(content=json_str, media_type='application/json', status_code=status.HTTP_200_OK)


@app.post("/init/")
async def init(request: Request):
    request_data = await request.json() 
   
    if 'userId' not in request_data:
        json_str = json.dumps({'error': 'Missing "user_id" field in JSON request'}, default=str)
        return Response(content=json_str, media_type='application/json', status_code=status.HTTP_400_BAD_REQUEST)
    
    user_id = request_data['userId']    
    with_stream = request_data['stream']

    decision_agent = decision_agent_factory.create()

    conversational_agent = conversational_agents_handler.initialize_by_user_id(user_id=user_id, decision_agent=decision_agent)

    if with_stream:
        answer_generator = conversational_agent.proactive_stream() 
        return StreamingResponse(answer_generator, media_type="text/event-stream;charset=UTF-8")
    else:
        answer = await conversational_agent.proactive_instruct() 
        return JSONResponse(content=answer, headers={"Content-Type": "application/json; charset=UTF-8"}) 


@app.post("/instruct/")
async def instruct(request: Request):
    request_data = await request.json() 

    if 'content' not in request_data:
        json_str = json.dumps({'error': 'Missing "content" field in JSON request'}, default=str)
        return Response(content=json_str, media_type='application/json', status_code=status.HTTP_400_BAD_REQUEST)
    
    if 'userId' not in request_data:
        json_str = json.dumps({'error': 'Missing "userId" field in JSON request'}, default=str)
        return Response(content=json_str, media_type='application/json', status_code=status.HTTP_400_BAD_REQUEST)
    
    user_id = request_data['userId']   
    instruction = request_data['content']
    with_stream = request_data['stream'] 

    decision_agent = decision_agent_factory.create()

    conversational_agent = conversational_agents_handler.get_by_user_id(user_id=user_id, decision_agent=decision_agent)

    if with_stream:
        answer_generator = conversational_agent.stream(instruction)    
        return StreamingResponse(answer_generator, media_type="text/event-stream;charset=UTF-8")
    else:
        answer = await conversational_agent.instruct(instruction) 
        return JSONResponse(content=answer, headers={"Content-Type": "application/json; charset=UTF-8"})   

    

