from string_art.core import StringArtGenerator
from string_art.algorithms.naive import NaiveAlgorithmConfig
from string_art.visualization import DefaultVisualizer

config = NaiveAlgorithmConfig()
generator = StringArtGenerator(config)

# load a [1, 400, 400] grayscale image with values between 0 (white) and 1 (black) assuming black strings on light background by default.
img = generator.load_image("data/inputs/cat_400.png")

# generate the string art reconstruction using the default configuration
reconstruction = generator.generate(img)

visualizer = DefaultVisualizer(config, generator.store)
visualizer.show_animation()