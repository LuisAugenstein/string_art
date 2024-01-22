import imageio.v3 as imageio
from string_art.config import get_default_config
import string_art

img = imageio.imread('data/inputs/cat.png')
config = get_default_config()
config.n_pins = 16
string_art.run(img, config)
