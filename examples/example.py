import imageio.v3 as imageio
from string_art.config import get_config
import string_art


img = imageio.imread('data/inputs/cat.png')
config = get_config()
config.name_of_the_run = 'test'
config.n_pins = 16
config.string_thickness = 0.15
config.use_cuda = False
config.loss_type = 'optimized-loss'
string_art.run(img, config)

"""
expexted output:
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
