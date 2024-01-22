# import os
# import ray
from tqdm import tqdm
from typing import Callable, TypeVar
# from ray.remote_function import RemoteFunction
# from string_art.config import get_config

T, S = TypeVar('T'), TypeVar('S')


# def __init_ray(f: Callable[[T], S], cpu_count: int) -> RemoteFunction:
#     if not ray.is_initialized():
#         print(f'Initializing Ray with {cpu_count} CPUs')
#         ray.init(num_cpus=os.cpu_count())

#     f_id = ray.put(f)

#     @ray.remote
#     def operation(data) -> S:
#         f = ray.get(f_id)
#         return f(data)

#     return operation


# def parallel_map(function: Callable[[T], S], data: list[T], use_tqdm=True) -> list[S]:
#     operation = __init_ray(function, cpu_count=os.cpu_count())
#     tasks = [operation.remote(x) for x in data]
#     if not use_tqdm:
#         return ray.get(tasks)

#     running = tasks
#     pbar = tqdm(total=len(running))
#     while len(running) > 0:
#         finished, running = ray.wait(running, num_returns=len(running), timeout=0.5)
#         pbar.update(len(finished))
#     pbar.close()
#     return ray.get(tasks)


def map(function: Callable[[T], S], data: list[T], use_tqdm=True, performance_mode=None) -> list[S]:
    # if performance_mode is None:
    #     performance_mode = get_config()
    # if performance_mode:
    #     return parallel_map(function, data, use_tqdm)
    return [function(x) for x in tqdm(data)] if use_tqdm else [function(x) for x in data]
