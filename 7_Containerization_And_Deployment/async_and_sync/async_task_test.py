import time
import asyncio

async def task():
    print("Start the sync task")
    await asyncio.sleep(5)
    print("After 5 seconds of sleep")


async def spawn_task():
    await asyncio.gather(task(),task(),task())

start_time = time.time()
asyncio.run(spawn_task())
duration = time.time() - start_time

print("Process completed in  : {}-seconds".format(duration))