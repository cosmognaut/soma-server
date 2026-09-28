import time
import numpy as np
from random import choice
from pydantic import BaseModel
from typing import Callable, Literal
from logic.graph import ollaya_router
from langchain_openai import ChatOpenAI

MODEL = ChatOpenAI(
    model="qwen2.5:3b",
    temperature=0,
    base_url="http://127.0.0.1:8080/v1",
    api_key="dummy"
)


QUERIES = [
        "Write a script to calculate the first 2000 digits of pi", # coding task
        "How are you?" # general task,
        "How do I find out the value of the 89th Fibonacci number?", # coding task again
        "What is the capital of France?" # general task again
]

def evaluate(function: Callable, iter: int = 50, discard: int = 3):
    # no accuracy measurement for now
    """
    Execute the given function and note how much time it took for `iter` iterations, and return a numpy array for execution times for `iter` iterations.
    
    Parameters:
        function: the function to be evaluated
        iter: how many iterations to run the function for; default is 50
        discard: how many iterations to discard so that JIT/weights do not skew the measurements; default is 3

    Returns:
        a numpy array containing execution times for `iter` iterations
    """
    exec_times = []
    for i in range(iter):
        start = time.perf_counter_ns()
        # call the function iter times
        query = choice(QUERIES) # random query
        function(query)
        end = time.perf_counter_ns()
        if i >= discard:
            exec_times.append(end-start)
    return np.array(exec_times)

class StructuredDecision(BaseModel):
    """Model for the supervisor node"""
    action: Literal['Coding', 'General'] # for vision, we inspect file_path manually; see logic/graph.py

NAIVE_DECISION_MODEL= MODEL.with_structured_output(StructuredDecision) # instantiate once

def generative_llm_router(query: str):
    """Routes a query through a standard locally running LLM"""
    result = NAIVE_DECISION_MODEL.invoke(input=f"Based on this message: {query} just reply using one word, either Coding or General related to whether the query is related to a coding task, or is a general query")
    return result.action

def main():
    percentiles = np.array([50, 90, 99])
    naive_times = evaluate(generative_llm_router)
    naive_p = np.percentile(naive_times, percentiles)
    print(f"For the naive method time taken for 50 iterations, with 3 discarded was:\n p50 - {naive_p[0]/1e6}ms\n p90 - {naive_p[1]/1e6}ms\n p99 - {naive_p[2]/1e6}")
    decision_times = evaluate(ollaya_router)
    decision_p = np.percentile(decision_times, percentiles)
    print(f"For the decision method time taken for 50 iterations, with 3 discarded was:\n p50 - {decision_p[0]/1e6}ms\n p90 - {decision_p[1]/1e6}ms\n p99 - {decision_p[2]/1e6}")

if __name__ == "__main__":
    print("Starting basic benchmarks...")
    main()
