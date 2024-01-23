import imageio.v3 as imageio
from string_art.config import get_config
import string_art


img = imageio.imread('data/inputs/cat.png')
config = get_config()
config.name_of_the_run = 'test'
config.n_pins = 32
config.string_thickness = 0.15
config.use_cpus = 4
string_art.run(img, config)
