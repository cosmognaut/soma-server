import uuid
import json
import time
import random
import asyncio
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

async def produce_chunks(history: list[AnyMessage], buffer: asyncio.Queue):
    """
    Move the chunks yielded from the graph into an async queue for consumption.

    Parameters:
        history - a list of AnyMessage objects, containing the user message history
        buffer - an asyncio.Queue containing chunks from the graph

    Returns:
        nothing
    """
    try:
        async for chunk in graph.start(history):
            await buffer.put(chunk)
            print(f"PUT CHUNK: {chunk}")
            print(time.perf_counter())
    except Exception as e:
        print(f"Exception: {e}")
    finally:
        await buffer.put("END")

async def consume_chunks(buffer: asyncio.Queue):
    """
    Consume the chunks currently in the async queue in the background.

    Parameters:
        buffer - an asyncio.Queue containing chunks from the graph

    Returns:
        nothing
    """
    # TODO: add error streaming later.
    verbs = ["crunching...", "distilling...", "working hard on it...", "connecting the dots...", "spinning up..."]
    while True:
        try:
            async with asyncio.timeout(10):
                chunk = await buffer.get()
                print(f"GET CHUNK: {chunk}")
                print(time.perf_counter())
                if chunk == "END":
                    break
        except asyncio.TimeoutError:
            print("The current chunk took more than 10 seconds to be received")
            chunk = {'type': "custom", 'data': {'status': f"{random.choice(verbs)}"}}
        try:
            if chunk['type'] == "updates" and isinstance(list(chunk['data'].values())[0]['messages'][0], AIMessage) and list(chunk['data'].values())[0]['messages'][0].content != '':
                response: AIMessage = list(chunk['data'].values())[0]['messages'][0]
                data = json.dumps(StreamChunk(type="answer", payload=str(response.content)).model_dump())
                print(data)
                yield f"data: {data}\n\n"
            elif chunk['type'] == "custom":
                data = json.dumps(StreamChunk(type="status", payload=str(chunk['data']['status'])).model_dump())
                print(data)
                yield f"data: {data}\n\n"
        except Exception as e:
            print(f"Exception: {e}")
            print(chunk)

@app.post("/api/message", response_class=StreamingResponse)
async def stream_message(message_request: MessageRequest) -> StreamingResponse:
    # message_dict = message_request.model_dump()
    history: list[AnyMessage] = []
    user_message: Message = message_request.input
    user_message.content += f"{message_request.file_name}"
    message_request.history.append(user_message)
    for message in message_request.history:
        if message.role == "user":
            history.append(HumanMessage(message.content))
        elif message.role == "assistant":
            history.append(AIMessage(message.content))
    buffer = asyncio.Queue()
    task = asyncio.create_task(produce_chunks(history, buffer))
    return StreamingResponse(consume_chunks(buffer))


# TODO: revamp to allow multiple uploads later
@app.post("/api/upload")
async def create_upload_file(file: UploadFile):
    identifier: str = f"{uuid.uuid1()}"
    extension: str = str(file.filename).split(".")[1]
    with open(f"./app/uploads/{identifier}.{extension}", "wb") as my_file:
        content = await file.read()
        my_file.write(content)
    return UploadResponse(file_id=identifier, file_name=f"{identifier}.{extension}").model_dump()
