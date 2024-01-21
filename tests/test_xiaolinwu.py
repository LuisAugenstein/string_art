import numpy as np
from scipy.io import loadmat
from string_art.transformations import xiaolinwu


def test_xiaolinwu():
    string = xiaolinwu(np.array([[4095, 2055], [7, 2055]]))
    string = xiaolinwu(np.array([[3, 6], [84, 59]]))
    string = np.array(string).T
    matlab_string = loadmat('tests/data/xiaolinwu_test_string.mat')['A']
    assert np.allclose(string, matlab_string)


test_xiaolinwu()
