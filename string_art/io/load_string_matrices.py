import os
from string_art.preprocessing import precompute_string_matrix, high_res_to_low_res_string_matrix
from string_art.io.root_path import root_path
from string_art.io.mkdir import mkdir
import torch


def load_string_matrices(n_pins: int, pin_side_length: float, string_thickness: float, min_angle: float, high_res: int, low_res: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns
    -
    A_high_res: torch.shape([high_res**2, n_strings]) coo sparse matrix with values between 0 and 1 indicate how much a pixel i is darkened if edge j is active.
    A_low_res: torch.shape([low_res**2, n_strings])   resized A_high_res with values between 0 and 1
    valid_edges_mask: torch.shape([n_strings])        binary mask for excluding edges from the optimization.
    """
    string_matrices_dir = f"{root_path}/data/string_matrices"
    config_dir = f'{string_matrices_dir}/{n_pins}_{pin_side_length}_{string_thickness}_{min_angle:.4f}_{high_res}_{low_res}'

    high_res_path, valid_edges_mask_path = f'{config_dir}/A_high_res.pt', f'{config_dir}/valid_edges_mask.pt'
    if os.path.exists(high_res_path) and os.path.exists:
        A_high_res = torch.load(high_res_path)
        valid_edges_mask = torch.load(valid_edges_mask_path)
    else:
        A_high_res, valid_edges_mask = precompute_string_matrix(n_pins, pin_side_length, string_thickness, min_angle, high_res)
        print('saving A_high_res to disk...')
        mkdir(config_dir)
        torch.save(A_high_res, high_res_path)
        torch.save(valid_edges_mask, valid_edges_mask_path)
    A_high_res = A_high_res.coalesce()

    low_res_path = f'{config_dir}/A_low_res.pt'
    if os.path.exists(low_res_path):
        A_low_res = torch.load(low_res_path)
    else:
        A_low_res = high_res_to_low_res_string_matrix(A_high_res, low_res)
        print('saving A_low_res to disk...')
        torch.save(A_low_res, low_res_path)
    A_low_res = A_low_res.coalesce()

    return A_high_res, A_low_res, valid_edges_mask
