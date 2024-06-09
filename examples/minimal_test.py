import torch

from string_art.optimization.iterative_greedy_optimizer import IterativeGreedyOptimizer
from string_art.optimization.losses.high_res_to_low_res_matrix import high_res_to_low_res_matrix
from string_art.optimization.losses.optimized_loss import OptimizedLoss


torch.set_default_dtype(torch.float64)
low_res = 2
high_res = 4
n_strings = 5
img = torch.tensor([[0.7, 0.],
                    [0.8, 0.6]])
importance_map = torch.ones_like(img)
rows = torch.tensor([0, 5, 9, 13,  8, 9, 10, 11,  3, 6, 10, 13,  0, 5, 6, 11,  3, 5, 6, 8])
cols = torch.tensor([0, 0, 0, 0,   1, 1, 1, 1,    2, 2, 2, 2,    3, 3, 3, 3,   4, 4, 4, 4])
values = torch.tensor([0.250, 0.125, 0.125, 0.250,
                       0.250, 0.125, 0.125, 0.250,
                       0.250, 0.125, 0.125, 0.250,
                       0.250, 0.5, 0.5, 0.250,
                       0.250, 0.5, 0.5, 0.250])
A_high_res = torch.sparse_coo_tensor(torch.stack([rows, cols]), values, size=(high_res**2, n_strings)).coalesce()
A_low_res = high_res_to_low_res_matrix(low_res, high_res) @ A_high_res
print(A_low_res.to_dense())

# OptimizedLoss(img, torch.ones_like(importance_map), A_high_res, A_low_res)
# optimizer = IterativeGreedyOptimizer(losses[config.loss_type](), StringSelection(valid_edges_mask), [
#                                      LoggingCallback(n_edges=A_high_res_csc.shape[1])], n_steps_max=config.n_steps)
# x = optimizer.optimize()
