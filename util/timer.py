from time import time

def timer(func):
    def wrapper(*args, **kwargs):
        t = time()
        val = func(*args, **kwargs)
        t = time() - t
        print(f"Time Taken: {t}s")
        return val

    return  wrapper