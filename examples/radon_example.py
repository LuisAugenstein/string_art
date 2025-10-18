import torch
from string_art import edges, pins
from string_art.core import StringArtGenerator
from string_art.algorithms.radon import RadonAlgorithmConfig
from string_art.core.default_visualizer import DefaultVisualizer

torch.set_default_dtype(torch.float64)

config = RadonAlgorithmConfig(
    n_pins=300,
    n_strings=7000,
    p_min=0.0001
)
generator = StringArtGenerator(config)

# load a [1, 400, 400] grayscale image with values between 0 (white) and 1 (black) assuming black strings on light background by default.
img = generator.load_image("data/inputs/cat_400.png")

# generate the string art reconstruction using the default configuration
reconstruction = generator.generate(img)

visualizer = DefaultVisualizer(config, generator.store)
visualizer.show_animation()

pins_angle_based = pins.angle_based(config.n_pins)
strings_index_based = edges.angle_to_index_based(pins_angle_based, reconstruction.strings)
strings_angle_based = edges.angle_based(pins_angle_based, strings_index_based)
assert torch.allclose(strings_angle_based, reconstruction.strings)
