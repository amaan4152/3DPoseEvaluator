from functools import wraps
import time


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        _t_start = time.time()
        out = func(*args, **kwargs)
        _t_end = time.time()

        _duration = _t_end - _t_start
        print(f"[DIAGNOSTICS]: Elapsed time -> {_duration}")
        return out

    return wrapper
