import numpy as np
import imageio.v3 as imageio
from scipy.io import loadmat
from tests.utils import measure_time
from string_art.transformations import matlab_imresize
from string_art.preprocessing import high_res_to_low_res_indices, high_res_to_low_res_indices_optimized


def test_matlab_imresize():
    """
    Compares the reimplemented matlab_imresize with the original matlab imresize function.
    """
    matlab_low_res_img = loadmat('tests/data/matlab_resized_cat_256.mat')['imOrig']
    img = imageio.imread('data/inputs/cat.png') / 255
    low_res_img = matlab_imresize(img, output_shape=(256, 256))
    assert np.allclose(low_res_img, matlab_low_res_img)
