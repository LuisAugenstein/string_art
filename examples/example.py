import imageio.v3 as imageio
import string_art

img = imageio.imread('data/inputs/cat.png')
config = string_art.get_config()
string_art.run(img, config)
