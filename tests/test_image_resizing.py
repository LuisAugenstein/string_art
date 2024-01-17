from string_art.transformations import imresize
import imageio.v3 as imageio
import numpy as np
from scipy.io import loadmat


def test_image_resize():
    img = imageio.imread('data/inputs/cat.png') / 255
    low_res_img = imresize(img, output_shape=(256, 256))
    matlab_low_res_img = loadmat('tests/data/matlab_resized_cat_256.mat')['imOrig']
    assert np.allclose(low_res_img, matlab_low_res_img)
