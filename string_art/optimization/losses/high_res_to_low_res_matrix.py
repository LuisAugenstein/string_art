import torch


def high_res_to_low_res_matrix(low_res: int, high_res: int) -> torch.Tensor:
    """
    matrix that scales a flattened high_res x high_res image down to a low_res x low_res image by averaging the pixels.

    example: high_res=6, low_res=2
      x_11 x_12 x_13 x_14 x_15, x_16           
      x_21 x_22 x_23 x_24 x_25, x_26
      x_31 x_32 x_33 x_34 x_35, x_36   =>  y_11  y_12     where y_11 = 1/9 sum(x_11 x_12 x_13 x_21 x_22 x_23 x_31 x_32 x_33)
      x_41 x_42 x_43 x_44 x_45, x_46   =>  y_21  y_22     where y_12 = 1/9 sum(x_14 x_15, x_16 x_24 x_25, x_26 x_34 x_35, x_36)
      x_51 x_52 x_53 x_54 x_55, x_56
      x_61 x_62 x_63 x_64 x_65, x_66                                

    Returns
    -
    correspondence_map: torch.shape([low_res**2, high_res**2])
    """
    super_sampling_factor = high_res // low_res
    n_pixels_high_res = high_res**2
    n_pixels_low_res = low_res**2

    rows = torch.arange(n_pixels_low_res).reshape(low_res, low_res)
    rows = rows.repeat_interleave(super_sampling_factor, dim=0).repeat_interleave(super_sampling_factor, dim=1).flatten()
    cols = torch.arange(n_pixels_high_res)
    values = torch.ones(n_pixels_high_res) / (super_sampling_factor ** 2)
    indices = torch.stack([rows, cols])
    return torch.sparse_coo_tensor(indices, values, size=(n_pixels_low_res, n_pixels_high_res)).coalesce()
