from string_art.transformations import imresize
import imageio.v3 as imageio
import numpy as np
from scipy.io import loadmat
from skimage.transform import resize


def test_image_resize():
    img = imageio.imread('data/inputs/cat.png') / 255
    low_res_img = imresize(img, output_shape=(256, 256))
    matlab_cat = loadmat('tests/data/matlab_resized_cat_256.mat')['imOrig']
    assert np.allclose(low_res_img, matlab_cat)

    # Note that scipy resize function does not produce the same result as matlab
    scipy_low_res_img = resize(img, (256, 256))
    assert not np.allclose(low_res_img, scipy_low_res_img)
