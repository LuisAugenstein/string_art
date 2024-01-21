import numpy as np

Edge = np.ndarray
"""np.shape([2], dtype=int) pin indices i,j represent abstract edge between the pins."""

String = tuple[np.ndarray, np.ndarray, np.ndarray]
"""shape([N, 3]) an anti-aliased line along a pixel grid. 
The first two columns contain the integer x,y coordinates of the pixels.
The third column contains the string intensity value of the pixel between 0 and 1."""
