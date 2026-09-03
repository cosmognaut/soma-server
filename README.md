<h1 align="center">soma v0.1</h1>
<p align="center">An agentic workbench for keeping your sensitive data off the cloud, while still giving you access to the latest and greatest open source models</p>

## What this is
This is the backend server and logic for SIH26117, a project me and my team have been working on. We have been calling this tool _Soma_ - it's an agentic workbench, specifically suited for industry workers that lets you access open source models with multimodal routing, while maintaining zero outbound connections. 

## Built with
The core workflow (inside `logic/`) was built using LangGraph, with OCR support from docling. We are currently using proprietary LLMs for fast development because of hardware issues. The model configured right now is `gemini-3.8-flash-high`.
The server itself uses FastAPI to expose the workflow.

## Test it locally
This is still **very** early in development, so you may notice some things breaking. But still, for development, follow the below steps:
1. Clone the repository:

## Limitations
1. Argubaly the biggest limitation is that we're using a proprietary model for testing right now. Things ARE going to break with the VLM + Coding LLM architecture in open source models. This is a severe limitation.
2. There is no "general" model yet - one can either run a coding task or a vision task, which are routed to the singular model we've been using. 
3. No multiple file uploads for now.
4. No server persistence. We are not using RAG for now, so persistence via Qdrant or something else doesn't make sense. The client captures the entire conversation history in their requests, so we're not even using LangGraph's checkpointers for saving memory.
5. This is a minor one but is still worth mentioning - we don't stream any errors to the client right now.

## Architecture
Still kind of undecided upon.
