import imageio.v3 as imageio
from string_art.config import get_config
import string_art
from math import pi

"""reproduces the cat example when using only 16 pins."""

img = imageio.imread('data/inputs/cat.png')
config = get_config()
config.name_of_the_run = 'test'
config.invert_input = True
config.invert_output = True
config.n_pins = 16
config.string_thickness = 0.15
config.frame_diameter = 614.4
config.pin_side_length = 2
config.super_sampling_window_width = 8
config.min_angle = pi / 8
config.plot_optimization = True
config.use_cuda = False
config.loss_type = 'optimized-loss'
config.n_steps = 5
string_art.run(img, config)

"""
expexted output:
1: edge-287  209430.5661613157717511
2: edge-280  209331.2226279406168032
3: edge-28   209233.6023791428015102
4: edge-84   209136.2302334080159198
5: edge-281  209038.9208334384893533

Original output of the Matlab version
1: edge-287  209430.5661613157426473
2: edge-280  209331.2226279405876994
3: edge-28   209233.6023791427724063
4: edge-84   209136.2302334079868160
5: edge-281  209038.9208334384602495
...
411: edge-472  183799.5702614040637854
412: edge-264  183773.8475774565595202
413: edge-64   183748.5604095249727834
414: edge-473  183723.4237057730206288
415: edge-265  183698.3423039136396255
416: edge-65   183673.6307157295523211
"""
