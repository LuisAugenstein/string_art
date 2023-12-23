from dataclasses import dataclass
from math import pi


@dataclass
class Config:
    input_file_path = 'input/cat.png'
    output_dir_path = 'output/cat'
    output_filename_prefix = 'cat'
    invert_input = True
    """false -> reconstruct white area, true -> reconstruct black area"""
    invert_output = True
    """false -> white string, true -> black string"""
    n_pins = 256
    string_thickness = 0.15
    """physical thickness of the thread in mm"""
    frame_diameter = 614.4
    """physical diameter of the circular frame in mm. ensure that resolution, i.e., frame_diameter / thread_thickness is a power of 2"""
    pin_side_length = 2
    """physical side length of a pin with quadratic cross section in mm"""
    super_sampling_window_width = 8
    """side length of the super sampling window must be a power of 2"""
    min_angle = pi / 8
    """Minimum angle (measured from frame center) between two connected pins"""
    data_path = './data'

    @property
    def high_resolution(self) -> int:
        """should be a power of 2. Set frame_diameter and string_thickness accordingly."""
        return self.frame_diameter // self.string_thickness

    @property
    def low_resolution(self) -> int:
        return self.high_resolution // self.super_sampling_window_width


def get_config() -> Config:
    return Config()
