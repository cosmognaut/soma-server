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

async def stream_with_heartbeat(history: list[AnyMessage]):
    """
    Stream currently ongoing events with heartbeat every ten seconds.

    Parameters:
        history - a list of AnyMessage objects. This is the user conversation history.
    Returns:
        nothing
    """
    iterator = graph.start(history)
    verbs = ["crunching...", "distilling...", "working hard on it...", "connecting the dots...", "spinning up..."]
    while True:
        try:
            async with asyncio.timeout(10):
                chunk = await anext(iterator)
            # if we are here, we have a chunk
            try:
                if chunk['type'] == "updates" and isinstance(list(chunk['data'].values())[0]['messages'][0], AIMessage) and list(chunk['data'].values())[0]['messages'][0].content != '':
                    response: AIMessage = list(chunk['data'].values())[0]['messages'][0]
                    data = StreamChunk(type="answer", payload=str(response.content)).model_dump_json()
                    print(data)
                    yield f"data: {data}\n\n"
                elif chunk['type'] == "custom":
                    data = json.dumps(StreamChunk(type="status", payload=str(chunk['data']['status'])).model_dump())
                    print(data)
                    yield f"data: {data}\n\n"
            except Exception as e:
                print(f"Exception: {e}")
                print(chunk)
        except TimeoutError:
            # 10 seconds have passed but the generator is still running
            status_message = random.choice(verbs)
            data = StreamChunk(type="status", payload=status_message).model_dump_json()
            yield f"data: {data}\n\n"

        except StopAsyncIteration:
            # generator has naturally finished executing. End the loop.
            break
        except Exception as e:
            # a real error has ocurred somewhere.
            yield f'data: {"error": "{str(e)}"}\n\n'
            break


@app.post("/api/message", response_class=StreamingResponse)
async def stream_message(message_request: MessageRequest) -> StreamingResponse:
    """
    Stream the graph's current status asynchronously.
    """
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
    return StreamingResponse(stream_with_heartbeat(history), media_type="text/event-stream")


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
