from functools import lru_cache
import torch
from typing import Callable

LineProfile = Callable[[torch.Tensor], torch.Tensor]
"""
Parameters
-
d: [N] distances from the center of the line

Returns
-
pixel_intensity: [N] pixel intensity at the given distance between 0=low_intensity and 1=high_intensity
"""


@lru_cache
def trapez(width: float, height: float, slope: float) -> LineProfile:
    """                            
    Parameters
    -
    width: width of the maximum trapez height
    height: maximum trapez height
    slope: how fast the slope should decline. an infinite slope equals a rectangular shape
    """
    def line_profile_trapez(d: torch.Tensor) -> torch.Tensor:
        """d: [N] distances (non-negative) from the center"""
        profile_height = torch.zeros_like(d)
        profile_height[d <= width] = height
        profile_height[d > width] = slope*(width - d[d > width]) + height
        return torch.clip(profile_height, 0, height)
    return line_profile_trapez
