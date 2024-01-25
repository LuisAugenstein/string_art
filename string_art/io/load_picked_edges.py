import os
import numpy as np
from string_art.optimization import StringOptimizer
from string_art.io.root_path import get_project_dir


def load_picked_edges(name_of_the_run: str, optimizer: StringOptimizer) -> np.ndarray:
    project_dir = get_project_dir(name_of_the_run)
    x_path = f'{project_dir}/x.npy'
    if name_of_the_run != 'test' and os.path.exists(x_path):
        return np.load(x_path)
    x = optimizer.optimize()
    np.save(x_path, x)
    return x
