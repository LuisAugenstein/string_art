import numpy as np
from string_art.optimization.greedy_multi_sampling_data_object_l2 import GreedyMultiSamplingDataObjectL2
from scipy.sparse import csr_matrix
from string_art.optimization.losses import SimpleLoss, OptimizedLoss


def optimize_strings_greedy_multi_sampling(img: np.ndarray, importance_map: np.ndarray, A_high_res: csr_matrix, A_low_res: csr_matrix, valid_edges_mask: np.ndarray):
    n_edge_decimals = len(str(A_high_res.shape[1]))
    # The SimpleLoss produces the same results as the OptimizedLoss, but is much slower.
    # loss = SimpleLoss(img, np.ones_like(importance_map), A_high_res, np.sqrt(A_low_res.shape[0]).astype(int))
    loss = OptimizedLoss(img, np.ones_like(importance_map), A_high_res, A_low_res)
    obj = GreedyMultiSamplingDataObjectL2(loss, valid_edges_mask)

    n_steps = 10000
    best_f_score = np.inf
    mode = 'add'
    switched = False

    for step in range(1, n_steps + 1):
        i_next_edge, f_score = obj.find_best_string(mode)
        if f_score is None or f_score >= best_f_score:
            print(f'{step}: edge-None')
            if switched:
                break
            switched = True
            mode = 'remove' if mode == 'add' else 'add'
            print(f'\n \tINFO: Switching mode to {mode} \n')
            continue

        switched = False
        print(f'{step}: edge-{str(i_next_edge) + ' ' * (n_edge_decimals - len(str(i_next_edge)))}  {f_score:16.16f}')
        obj.choose_string_and_update(i_next_edge, mode)
        best_f_score = f_score

    return obj.x, obj.picked_edges_sequence
