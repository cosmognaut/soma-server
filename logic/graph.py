import os
import torch
import docker
import asyncio
import operator
from dotenv import load_dotenv
from pydantic import BaseModel
from .hack import ocr_required
from langchain.tools import tool
from typing import Any, Optional
from langgraph.types import Command
from langgraph.config import get_stream_writer
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from langchain.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage


load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") - not used right now
MY_ENDPOINT = os.getenv("MY_ENDPOINT")
ENDPOINT_KEY = os.getenv("API_KEY")

torch.set_num_threads(2)

MODEL = ChatOpenAI(
    model="gemini-3.8-flash-high",
    base_url=MY_ENDPOINT,
    api_key=ENDPOINT_KEY,
    temperature=0
)

## Coding Subgraph
# Define tools
@tool
def run_python(python_code: str):
    """Runs python code in an isolated container and returns the output"""
    # spin up an ephemeral container
    writer = get_stream_writer()
    writer({"status": "running python code..."})
    client = docker.from_env()
    try:
        output = client.containers.run(image="python:3.12-slim", command=['timeout', '5', 'python', '-c', f'{python_code}'], detach=False, remove=True)
        output = output.decode('utf-8')
    except docker.errors.ContainerError as e:
        if e.exit_status == 124:
            output = "Timeout error [124] - please check that your code doesn't contain any infinite loops or excess recursion."
        else:
            output = f"There might be an error with your program: {e.stderr.decode('utf-8')}"
    return output

TOOLS_DICT = {
        'run_python': run_python
        }

class CodingState(TypedDict):
    """Coding Subgraph's internal state"""
    # shared state key with parent
    messages: Annotated[list[AnyMessage], operator.add]

def llm_node(state: CodingState):
    """Node for the LLM with tools, it returns a message here"""
    # this might not be needed now that I have implemented an async loop
    writer = get_stream_writer()
    if len(state["messages"]) <= 2:
        writer({"status": "asking the coding model..."})
    else:
        writer({"status": "coding model is on it..."})
    latest_message = state["messages"][-1]
    # print(f"[CODING_SUBGRAPH {llm_node.__name__} -> latest message is: {latest_message}]")
    model_with_tools = MODEL.bind_tools([run_python])
    system = SystemMessage(content="You are a coding agent. For any coding task, you must call the run_python tool to write and execute the code before giving your final answer - never just describe or print code without running it.")
    # invoke it with all messages so that it remembers everything.
    return {"messages": [model_with_tools.invoke(input=[system] + state["messages"])]}

def tool_node(state: CodingState):
    """Node for the tool execution. We check the last message for tool_calls. If there are indeed calls, we execute the tools and return a ToolMessage."""
    writer = get_stream_writer()
    writer({"status": "calling tool for secure sandboxed execution..."})
    # print(f"[CODING_SUBGRAPH {tool_node.__name__}]")
    last_message_tool_calls = state["messages"][-1].tool_calls
    # print(last_message_tool_calls)
    tool_outputs = []
    for tool_call in last_message_tool_calls:
        func_name = tool_call['name']
        func_args = tool_call['args']
        function = TOOLS_DICT[func_name]
        output = function.invoke(input=func_args)
        tool_outputs.append(ToolMessage(content=output, tool_call_id=tool_call['id']))
    return {"messages": tool_outputs}

def routing_function(state: CodingState):
    """Routing function that decides whether we should go to the tool node or not"""
    # still a work in progress
    writer = get_stream_writer()
    writer({"status": "seeing if I need to call a tool..."})
    # print(f"[CODING_SUBGRAPH {routing_function.__name__}]")
    last_message = state["messages"][-1]
    return last_message.tool_calls != []

coding_graph_builder = StateGraph(CodingState)
coding_graph_builder.add_node("llm_node", llm_node)
coding_graph_builder.add_node("tool_node", tool_node)
coding_graph_builder.add_edge(START, "llm_node")
coding_graph_builder.add_conditional_edges("llm_node", routing_function, {True: "tool_node", False: END})
coding_graph_builder.add_edge("tool_node", "llm_node")
coding_subgraph = coding_graph_builder.compile()

## Vision/Image Subgraph
class VisionState(TypedDict):
    """State schema for the vision graph"""
    file_path: str
    extracted_info: str
    # shared state key with parent
    messages: Annotated[list[AnyMessage], operator.add]

# base model for extraction output
class ExtractedData(BaseModel):
    document_type: str
    key_findings: list[str]
    requires_action: bool # will lead to an interrupt

async def parse_document(state: VisionState):
    """Node for parsing a document in the vision graph. This is a blocking function."""
    writer = get_stream_writer()
    writer({"status": "asking the vision model..."})
    # print(f"[VISION_SUBGRAPH parse_document]: Parsing document.., last message: {state["messages"][-1]}")
    # print(f"[VISION_SUBGRAPH parse_document]: Document path: {source}")
    source = f"{os.getcwd()}{state["file_path"]}"
    print(f"File souce is {source}")
    print(f"Path exists? {os.path.exists(source)}")
    ocr = ocr_required(source)
    print(f"OCR: {ocr}")
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=PdfPipelineOptions(do_ocr=ocr)
            ),
        }
    )
    current_status = {"status": "performing OCR..."} if ocr else {"status": "sifting through this document..."}
    writer = get_stream_writer()
    writer(current_status)
    # this is going to block the fucking event loop
    # let me allocate this another thread
    result = await asyncio.to_thread(converter.convert, source)
    return { "extracted_info" : result.document.export_to_markdown() }

def extract_and_describe_entities(state: VisionState):
    """Node for extracting and describing entities in the vision graph"""
    # we take the markdown from the state
    writer = get_stream_writer()
    writer({"status": "extracting entities..."})
    # print(f"[VISION_SUBGRAPH extract_entities]: Extracting entities.., last message: {state["messages"][-1]}")
    markdown = state["extracted_info"]
    # structured_model = MODEL.with_structured_output(ExtractedData)
    result = MODEL.invoke(input=f"Based on this markdown: {markdown} give a summary of the extracted text to the user. The user had given a document which was parsed to the markdown given to you.")
    # print(f"[VISION_SUBGRAPH extract_entities]: Extracted data: {result}")
    return { "extracted_info": result, "messages": [result] }

# first parse the document, then extract relevant entities.
vision_graph_builder = StateGraph(VisionState)
vision_graph_builder.add_node("parse_document", parse_document)
vision_graph_builder.add_node("extract_and_describe_entities", extract_and_describe_entities)
vision_graph_builder.add_edge(START, "parse_document")
vision_graph_builder.add_edge("parse_document", "extract_and_describe_entities")
vision_graph_builder.add_edge("extract_and_describe_entities", END)
vision_subgraph = vision_graph_builder.compile()

## Finally, the parent graph
class ParentState(TypedDict):
    """State for the parent graph"""
    messages: Annotated[list[AnyMessage], operator.add]
    file_path: Optional[str]

# model for the output for this model
class SupervisorModel(BaseModel):
    """Model for the supervisor_node"""
    action: str
    file_path: str

def supervisor_node(state: ParentState):
    """Supervisor node which decides whether to route to coding subgraph or the vision subgraph based on the user's prompt"""
    writer = get_stream_writer()
    writer({"status": "thinking..."})
    # print(f"[{supervisor_node.__name__}]")
    latest_message = state["messages"][-1]
    model = MODEL.with_structured_output(SupervisorModel)
    result = model.invoke(input=f"Based on this message: {latest_message} just reply using one word for action and another for the file path, if present - either Coding or Vision based on whether the task is related to coding or vision operations. If it's related to neither, just put None in action. If it's a task related to documents (i.e. vision), make sure that the file_path has the correct file path. Because this is running on a server, make sure that filepaths are formatted well, for example if the user gives abc.pdf you must write /app/uploads/abc.pdf for file path. Example of coding tasks: anything where code has to be written. Example of vision tasks: document parsing, ex. extracting entitites from a document, document path, etc.")
    # final_output = result.content[0]['text'] # Coding, Image or None
    # print(f"[{supervisor_node.__name__}]: {result}")
    # return a command object that routes to either the coding subgraph or the image subgraph
    print(f"Result is {result}")
    if result.action == "Coding":
        writer = get_stream_writer()
        writer({"status": "routing to the coding model..."})
        return Command(update={"messages": [SystemMessage(content="making sure I don't sudo rm -rf / this server...")]}, goto="coding_subgraph_node")
    if result.action == "Vision":
        # I am using a shared state key to share state between ParentState and VisionState
        # an alternative would be to call the subgraph right here, right now.
        writer = get_stream_writer()
        writer({"status": "routing to the vision model..."})
        return Command(update={"messages": [SystemMessage(content="deconstructing this data...")], "file_path": result.file_path}, goto="vision_subgraph_node")
    if result.action == "None":
        # there is no separate subgraph for the generall model, the general model is the one that classifies the task here.
        writer = get_stream_writer()
        writer({"status": "routing to the general model..."})
        return Command(update={"messages": [SystemMessage(content="getting ready for your query..")]}, goto="general_node")

def general_node(state: ParentState):
    """Node for general queries, that don't require a specialised coding or vision model"""
    # currently, this uses as the same model which is routing to coding or vision models.
    writer = get_stream_writer()
    writer({"status": "getting ready..."})
    system = SystemMessage(content="You are Soma, a helpful AI assistant developed by team Malai Chaap. You will help the user with their general queries.")
    return {"messages": [MODEL.invoke(input=[system] + state["messages"])]}

parent_graph_builder = StateGraph(ParentState)
parent_graph_builder.add_node("supervisor_node", supervisor_node)
parent_graph_builder.add_node("coding_subgraph_node", coding_subgraph)
parent_graph_builder.add_node("vision_subgraph_node", vision_subgraph)
parent_graph_builder.add_node("general_node", general_node)
parent_graph_builder.add_edge(START, "supervisor_node")
parent_graph_builder.add_edge("supervisor_node", END)
parent_graph = parent_graph_builder.compile()

async def start(messages: list[AnyMessage]):
    """
    Starts the graph workflow. This is a generator.

    Parameters:
        messages: list[AnyMessage] - list of messages for history

    Yields:
        chunk - a chunk dictionary to be cast into StreamChunk later
    """
    async for chunk in parent_graph.astream(
        {"messages" : messages},
        stream_mode=["updates", "custom"],
        version="v2",
        subgraphs=True
    ):
        yield chunk

if __name__ == "__main__":
    user_input = input("Ask: ")

    # this is the synchronous version
    for chunk in parent_graph.stream(
        {"messages" : [HumanMessage(content=user_input)]},
        stream_mode=["updates", "custom"],
        version="v2",
        subgraphs=True
    ):
        print(chunk)
    # result = parent_graph.invoke({"messages" : [HumanMessage(content=user_input)]})
    # last_message = result["messages"][-1]
    # print("===== FINAL MESSAGE =====")
    # try:
    #     if isinstance(last_message.content, str):
    #         print(last_message.content) # openai format
    #     else:
    #         print(last_message.content[0]['text']) # gemini format iirc
    # except Exception as e:
    #     print(f"Error printing message: {e}")
    #     print(last_message)
    #
