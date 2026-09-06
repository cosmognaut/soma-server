import uuid
import json
import time
import random
import asyncio
import aiofiles
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
    Move the chunks yielded from the graph into the async queue for consumption. Put a sentinel value "END" in the bfufer when the graph has been fully iterated over.

    Parameters:
        history - a list of AnyMessage objects, containing the user message history
        buffer - an asyncio.Queue containing chunks received from the graph

    Yields:
        nothing

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

    Yields:
        status updates for the client in form of SSEs

    Returns:
        nothing
    """
    verbs = ["crunching...", "distilling...", "working hard on it...", "connecting the dots...", "spinning up..."]
    while True:
        try:
            async with asyncio.timeout(10):
                chunk = await buffer.get()
                print(f"GET CHUNK: {chunk}")
                print(time.perf_counter())
                if chunk == "END": # sentinel value
                    break
        except asyncio.TimeoutError:
            print("The current chunk took more than 10 seconds to be received.")
            chunk = {"type": "custom", "data": {"status": f"{random.choice(verbs)}"}} 
        # if we're here we either have a real chunk or a >=10s heartbeat chunk

        try:
            if chunk['type'] == "updates" and isinstance(list(chunk['data'].values())[0]['messages'][0], AIMessage) and list(chunk['data'].values())[0]['messages'][0].content != '':
                response: AIMessage = list(chunk['data'].values())[0]['messages'][0]
                data = StreamChunk(type="answer", payload=str(response.content)).model_dump_json()
                print(data)
                yield f"data: {data}\n\n"
            elif chunk['type'] == "custom":
                data = StreamChunk(type="status", payload=str(chunk['data']['status'])).model_dump_json()
                print(data)
                yield f"data: {data}\n\n"
        except Exception as e:
            print(f"Exception: {e}")
            print(chunk)


@app.post("/api/message", response_class=StreamingResponse)
async def stream_message(message_request: MessageRequest) -> StreamingResponse:
    """
    Stream the graph's current status asynchronously.
    """
    # message_dict = message_request.model_dump()
    history: list[AnyMessage] = []
    user_message: Message = message_request.input
    user_message.content += f"\n File: {message_request.file_name}"
    message_request.history.append(user_message)
    for message in message_request.history:
        if message.role == "user":
            history.append(HumanMessage(message.content))
        elif message.role == "assistant":
            history.append(AIMessage(message.content))
    buffer = asyncio.Queue()
    task = asyncio.create_task(produce_chunks(history, buffer))
    return StreamingResponse(consume_chunks(buffer), media_type="text/event-stream")


@app.post("/api/upload")
async def create_upload_file(files: list[UploadFile]) -> list[UploadResponse]:
    """
    Upload multiple files to the server in chunks of 64KB, asynchronously.
    """
    responses = []
    for file in files:
        SIXTY_FOUR_KB = 65536
        identifier: str = f"{uuid.uuid4()}"
        extension: str = str(file.filename).split(".")[1]
        async with aiofiles.open(f"./app/uploads/{identifier}.{extension}", "wb") as my_file:
            content = await file.read(SIXTY_FOUR_KB)
            while content:
                await my_file.write(content)
                content = await file.read(SIXTY_FOUR_KB)
        responses.append(UploadResponse(file_id=identifier, file_name=f"{identifier}.{extension}"))
    return [response.model_dump() for response in responses]
