import torch
from string_art import edges, pins

def test_edges_conversion():
    n_pins = 8
    pins_angle_based = pins.angle_based(n_pins)
    edges_index_based = edges.index_based(n_pins)
    edges_angle_based = edges.angle_based(pins_angle_based, edges_index_based)
    edges_index_based_reconstructed = edges.angle_to_index_based(pins_angle_based, edges_angle_based)
    assert torch.allclose(edges_index_based, edges_index_based_reconstructed)

    edges_radon_parameter_based = edges.angle_to_radon_parameter_based(edges_angle_based)
    edges_angle_based_from_radon = edges.radon_parameter_to_angle_based(edges_radon_parameter_based)
    assert torch.allclose(edges_angle_based, edges_angle_based_from_radon, atol=1e-5)

