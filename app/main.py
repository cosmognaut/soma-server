import uuid
import json
from logic import graph
from collections.abc import AsyncIterable
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from .models import Message, MessageRequest, StreamChunk, UploadResponse
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage

app = FastAPI()

origins = [ 
    "*" # allow all for testing
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/message", response_class=StreamingResponse)
async def stream_message(message_request: MessageRequest) -> AsyncIterable[str]:
    # message_dict = message_request.model_dump()
    history: list[AnyMessage] = []
    message_request.history.append(message_request.input)
    for message in message_request.history:
        if message.role == "user":
            history.append(HumanMessage(message.content))
        elif message.role == "assistant":
            history.append(AIMessage(message.content))
    async for chunk in graph.start(history):
        try:
            if chunk['type'] == "updates" and isinstance(list(chunk['data'].values())[0]['messages'][0], AIMessage) and list(chunk['data'].values())[0]['messages'][0].content != '':
                response: AIMessage = list(chunk['data'].values())[0]['messages'][0]
                data = json.dumps(StreamChunk(type="answer", payload=str(response.content)).model_dump())
                yield f"data: {data}\n\n"
            elif chunk['type'] == "custom":
                data = json.dumps(StreamChunk(type="status", payload=str(chunk['data']['status'])).model_dump())
                yield f"data: {data}\n\n"
        except TypeError:
            print(chunk)
        # TODO: add error streaming later.

# TODO: revamp to allow multiple uploads later
@app.post("/api/upload")
async def create_upload_file(file: UploadFile):
    identifier: str = f"{uuid.uuid1()}"
    extension: str = str(file.filename).split(".")[1]
    with open(f"./uploads/{identifier}.{extension}", "wb") as my_file:
        content = await file.read()
        my_file.write(content)
    return UploadResponse(file_id=identifier, file_name=f"{identifier}.{extension}").model_dump()

