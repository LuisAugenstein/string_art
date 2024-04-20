import os
import torch
import numpy as np
from string_art.optimization import StringOptimizer
from string_art.io.root_path import get_project_dir


def load_picked_edges(name_of_the_run: str, optimizer: StringOptimizer) -> torch.Tensor:
    project_dir = get_project_dir(name_of_the_run)
    x_path = f'{project_dir}/x.pt'
    if name_of_the_run != 'test' and os.path.exists(x_path):
        return torch.load(x_path)
    x = optimizer.optimize()
    torch.save(x, x_path)
    return x
