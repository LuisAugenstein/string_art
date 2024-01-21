import time
from typing import Callable, TypeVar

T = TypeVar('T')


def measure_time(f: Callable[[None], T]) -> tuple[T, float]:
    start = time.time()
    result = f()
    t = time.time() - start
    return result, t
