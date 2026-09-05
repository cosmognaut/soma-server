# The Producer (asyncio.create_task):
# Run an async worker task in the background. Its sole job is to iterate through graph.astream(...) and push every chunk into an asyncio.Queue. When the graph finishes, it pushes a sentinel value (e.g. None or object()) into the queue. If it crashes, it pushes the exception.
# Documentation: https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task
#
# The Buffer (asyncio.Queue):
# Acts as the thread-safe/task-safe boundary between the graph execution and the HTTP response stream.
# Documentation: https://docs.python.org/3/library/asyncio-queue.html
#
# The Consumer (Your SSE Generator):
# A simple while True loop that pulls items from queue.get(). You wrap only the queue.get() call in an asyncio.timeout or asyncio.wait_for.
# Documentation: https://docs.python.org/3/library/asyncio-task.html#asyncio.wait_for
#
# How it behaves in practice
# During OCR: The producer is suspended waiting on await asyncio.to_thread(docling_run). The queue is empty.
# Every 15 seconds: queue.get() in the consumer times out. The consumer catches the timeout, yields : ping\n\n, and loops right back to waiting on the queue. The producer is completely unaffected.
# When OCR finishes and LLM starts: The producer dumps tokens into the queue. The consumer immediately pops them, formats the SSE message, and yields them to the client.
import time
import asyncio

queue = asyncio.Queue()
async def producer():
    await asyncio.to_thread(time.sleep, 5)
    tokens = "apple banana orange"
    await queue.put(tokens)

async def consumer():
    while True:
        try:
            async with asyncio.timeout(1):
                result = await queue.get()
                print(result)
                break
        except asyncio.TimeoutError:
            print("ping")

async def main():
    # blocking task
    task = asyncio.create_task(producer())
    await consumer()

if __name__ == "__main__":
    asyncio.run(main())
