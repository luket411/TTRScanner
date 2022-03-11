from sys import path
from os import path as ospath
path.append(ospath.join(ospath.dirname(__file__), ".."))

from time import time

from util.constants import ENABLE_TIMER

def timer(func):
    def wrapper(*args, **kwargs):
        if ENABLE_TIMER:
            t = time()
            val = func(*args, **kwargs)
            t = time() - t
            print(f"Time Taken: {t}s")
        else:
            val = func(*args, **kwargs)
        return val

    return  wrapper