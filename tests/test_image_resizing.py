from string_art.transformations import imresize
import imageio.v3 as imageio
import numpy as np
from scipy.io import loadmat
import time
from skimage.draw import line
from scipy.sparse import find


def test_matlab_imresize():
    img = imageio.imread('data/inputs/cat.png') / 255
    start = time.time()
    low_res_img = imresize(img, output_shape=(256, 256))
    t = time.time() - start
    matlab_low_res_img = loadmat('tests/data/matlab_resized_cat_256.mat')['imOrig']
    np.save('tests/data/matlab_resized_cat_256.npy', low_res_img)
    assert np.allclose(low_res_img, matlab_low_res_img)


def test_gpu_imresize():
    def draw_line(start, end, high_res, low_res):
        img = np.zeros((high_res, high_res))
        rr, cc = line(start[0], start[1], end[0], end[1])
        img[rr, cc] = 1.
        start = time.time()
        low_res_img = imresize(img, output_shape=(low_res, low_res))
        t = time.time() - start
        i, j, v = find(low_res_img)
        return np.stack([i, j]).T, v, t

    def get_bounding_box(start, end, high_res, low_res, margin=1):
        bbox_start = np.array([min(start[0], end[0]), min(start[1], end[1])])
        bbox_end = np.array([max(start[0], end[0]), max(start[1], end[1])])
        scale = high_res // low_res
        bbox_start = bbox_start - bbox_start % scale - margin*scale
        bbox_end = bbox_end - bbox_end % scale + (margin+1)*scale
        return np.clip(bbox_start, 0, high_res-1), np.clip(bbox_end, 0, high_res-1)

    def draw_line2(start, end, high_res, low_res):
        scale = high_res // low_res
        bbox_start, bbox_end = get_bounding_box(start, end, high_res, low_res, margin=2)
        high_res = np.max(bbox_end - bbox_start)
        if high_res % scale != 0:
            high_res += scale - high_res % scale
        string, v, t = draw_line(start-bbox_start, end-bbox_start, high_res, low_res=high_res//scale)
        string = bbox_start[None, :] // scale + string
        return string, v, t

    high_res, low_res = 1024, 256
    start, end = np.random.randint(0, high_res-1, size=2), np.random.randint(0, high_res-1, size=2)
    string1, v1, t1 = draw_line(start, end, high_res, low_res)
    string2, v2, t2 = draw_line2(start, end, high_res, low_res)
    assert np.allclose(string1, string2) and np.allclose(v1, v2)
