
from typing import Annotated
from fastapi import Depends, FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import json
import requests

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


# @app.post("/instruct/")
# async def instruct(request: Request):
#     request_data = await request.json() 

#     if 'content' not in request_data:
#         json_str = json.dumps({'error': 'Missing "content" field in JSON request'}, default=str)
#         return Response(content=json_str, media_type='application/json', status_code=status.HTTP_400_BAD_REQUEST)
    
#     if 'userId' not in request_data:
#         json_str = json.dumps({'error': 'Missing "userId" field in JSON request'}, default=str)
#         return Response(content=json_str, media_type='application/json', status_code=status.HTTP_400_BAD_REQUEST)
    
#     user_id = request_data['userId']   
#     instruction = request_data['content']
#     with_stream = request_data['stream'] 

#     decision_agent = decision_agent_factory.create()

#     conversational_agent = conversational_agents_handler.get_by_user_id(user_id=user_id, decision_agent=decision_agent)

#     if with_stream:
#         answer_generator = conversational_agent.stream(instruction)    
#         return StreamingResponse(answer_generator, media_type="text/event-stream;charset=UTF-8")
#     else:
#         answer = await conversational_agent.instruct(instruction) 
#         return JSONResponse(content=answer, headers={"Content-Type": "application/json; charset=UTF-8"})

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
    with_stream = request_data.get('stream', False)
    
    
    use_middleware = request_data.get('use_middleware', True)
    
    if not use_middleware:
        decision_agent = decision_agent_factory.create()
        conversational_agent = conversational_agents_handler.get_by_user_id(user_id=user_id, decision_agent=decision_agent)
        
        if with_stream:
            answer_generator = conversational_agent.stream(instruction)
            return StreamingResponse(answer_generator, media_type="text/event-stream;charset=UTF-8")
        else:
            answer = await conversational_agent.instruct(instruction)
            return JSONResponse(content=answer, headers={"Content-Type": "application/json; charset=UTF-8"})
    
    
    try:
        middleware_url = "http://localhost:5010/whisper"
        headers = {'Content-Type': 'application/json'}
        
        middleware_payload = {
            "message": instruction,
            "user_id": user_id,
            "stream": with_stream
        }
        
        print(f"Framework calling middleware: {middleware_payload}")
        
        if with_stream:
            response = requests.post(middleware_url, headers=headers, json=middleware_payload, stream=True)
            
            if response.status_code == 200:
                print("Framework middleware streaming success")
                
                def forward_stream():
                    for line in response.iter_lines():
                        if line:
                            yield line.decode('utf-8') + '\n'
                
                return StreamingResponse(forward_stream(), media_type="text/event-stream;charset=UTF-8")
            else:
                print(f"Framework middleware streaming failed: {response.status_code}")
                decision_agent = decision_agent_factory.create()
                conversational_agent = conversational_agents_handler.get_by_user_id(user_id=user_id, decision_agent=decision_agent)
                answer_generator = conversational_agent.stream(instruction)
                return StreamingResponse(answer_generator, media_type="text/event-stream;charset=UTF-8")
        else:
            response = requests.post(middleware_url, headers=headers, json=middleware_payload)
            
            if response.status_code == 200:
                print("Framework middleware non-streaming success")
                answer = response.json()
                return JSONResponse(content=answer, headers={"Content-Type": "application/json; charset=UTF-8"})
            else:
                print(f"Framework middleware non-streaming failed: {response.status_code}")
                decision_agent = decision_agent_factory.create()
                conversational_agent = conversational_agents_handler.get_by_user_id(user_id=user_id, decision_agent=decision_agent)
                answer = await conversational_agent.instruct(instruction)
                return JSONResponse(content=answer, headers={"Content-Type": "application/json; charset=UTF-8"})
            
    except Exception as e:
        print(f"Framework exception: {e}")
        decision_agent = decision_agent_factory.create()
        conversational_agent = conversational_agents_handler.get_by_user_id(user_id=user_id, decision_agent=decision_agent)
        
        if with_stream:
            answer_generator = conversational_agent.stream(instruction)
            return StreamingResponse(answer_generator, media_type="text/event-stream;charset=UTF-8")
        else:
            answer = await conversational_agent.instruct(instruction)
            return JSONResponse(content=answer, headers={"Content-Type": "application/json; charset=UTF-8"})
