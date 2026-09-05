import asyncio
import time

# goal: yield '{'updates': 'whatever'}' type of strings every 10s when a blocking function is running

def blocking(secs: int = 60) -> str:
    """
    A generic blocking function that blocks a thread.

    Parameters:
        secs - an integer value for the number of seconds this is going to block

    Returns:
        a simple string like "hello"
    """
    print("starting blocking task..")
    time.sleep(secs)
    print("sorry, it took some time")

async def background():
    """
    Loop in the background forever

    Parameters:
        none
    Returns:
        none
    """
    while True:
        print("background task is running")
        await asyncio.sleep(5)

async def main():
    # blocking_coro = asyncio.to_thread(blocking)
    # execute the blocking function independently
    task = asyncio.create_task(background())
    coro = asyncio.to_thread(blocking, 20)
    await coro

if __name__ == "__main__":
    asyncio.run(main())

