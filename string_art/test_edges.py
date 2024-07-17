import torch
import numpy as np
import string_art.edges as edges


def test_radon_index_based():
    torch.set_default_dtype(torch.float64)
    edges_angle_based = torch.tensor([[0, torch.pi/2],
                                      [0, torch.pi],
                                      [0, 3*torch.pi/2],
                                      [torch.pi/2, torch.pi],
                                      [torch.pi/2, 3*torch.pi/2],
                                      [torch.pi, 3*torch.pi/2]])
    s_target = torch.tensor([np.sqrt(2) / 2,
                            0,
                            -np.sqrt(2) / 2,
                            np.sqrt(2) / 2,
                            0,
                            - np.sqrt(2) / 2])
    alpha_target = torch.tensor([np.pi / 4,
                                np.pi / 2,
                                3 * np.pi / 4,
                                3 * np.pi / 4,
                                0,
                                np.pi / 4])

    s, alpha = edges.radon_parameter_based(edges_angle_based).T
    assert torch.allclose(s, s_target)
    assert torch.allclose(alpha, alpha_target)
