import time

def measure_latency(func, *args, **kwargs):
    start = time.time()
    output = func(*args, **kwargs)
    end = time.time()
    return output, end - start
