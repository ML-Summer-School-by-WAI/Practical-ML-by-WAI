import time


def task():
    print("Start the sync task")
    time.sleep(5)
    print("After 5 seconds of sleep")


start_time = time.time()
for i in range(3):
    task()

duration = time.time() - start_time

print("Process completed in  : {}-seconds".format(duration))