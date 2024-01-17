import imageio.v3 as imageio
import string_art
from math import pi

img = imageio.imread('data/inputs/cat.png')
config = string_art.get_config()
config.n_pins = 256
config.pin_side_length = 2
config.string_thickness = 0.15
config.frame_diameter = 307.2
config.min_angle = pi / 4
config.super_sampling_window_width = 4
string_art.run(img, config, 'cat')
