from contextlib import contextmanager
from sys import path
from os import path as ospath
path.append(ospath.join(ospath.dirname(__file__), ".."))

from time import time

from util.constants import ENABLE_TIMER

def timer(func):
    def wrapper(*args, **kwargs):
        if ENABLE_TIMER:
            t_start = time()
            val = func(*args, **kwargs)
            t_end = time()
            t = t_end - t_start
            print(f"Time Taken for function {ospath.basename(func.__code__.co_filename)}@{func.__code__.co_firstlineno}:{func.__name__}: {t}:s (Start time: {t_start}, End time: {t_end})")
        else:
            val = func(*args, **kwargs)
        return val

    return  wrapper


@contextmanager
def timer_context():
    start = time()
    yield
    end = time()
    print(f"Time taken for code snippet: {end-start}s")