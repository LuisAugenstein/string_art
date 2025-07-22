from dataclasses import dataclass
from string_art.core.string_art_config import StringArtConfig


@dataclass
class RadonAlgorithmConfig(StringArtConfig):
    n_radon_angles: int = 300
    residual_threshold: float = 0.01
    line_darkness: float = 0.018
    p_min: float = 0.00008
    t_start: float = 0.0000005
    t_end: float = 0.000065