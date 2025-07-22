from string_art.core import StringArtGenerator
from string_art.algorithms.radon import RadonAlgorithmConfig
from string_art.visualization import DefaultVisualizer

config = RadonAlgorithmConfig(
    n_pins=240,
    n_strings=5000,
)
generator = StringArtGenerator(config)

# load a [1, 400, 400] grayscale image with values between 0 (white) and 1 (black) assuming black strings on light background by default.
img = generator.load_image("data/inputs/cat_400.png")

# generate the string art reconstruction using the default configuration
reconstruction = generator.generate(img)

visualizer = DefaultVisualizer(config, generator.store)
visualizer.show_animation()