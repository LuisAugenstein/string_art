from dataclasses import dataclass
from math import pi


@dataclass
class Config:
    invert_input = True
    """false -> reconstruct white area, true -> reconstruct black area"""
    invert_output = True
    """false -> white string, true -> black string"""
    n_pins = 256
    """number of pins for the image reconstruction. assumed to be power of 2."""
    string_thickness = 0.15
    """physical thickness of the thread in mm"""
    frame_diameter = 614.4
    """physical diameter of the circular frame in mm. ensure that resolution, i.e., frame_diameter / thread_thickness is a power of 2"""
    pin_side_length = 2
    """physical side length of a pin with quadratic cross section in mm"""
    super_sampling_window_width = 8
    """side length of the super sampling window must be a power of 2"""
    min_angle = pi / 8
    """Minimum angle between two vectors from center to two pins. 
       Prevents strings from getting spanned between two pins that are very close. 
       A lower min_angle decreases performance since the optimizer has to consider more possible string connections."""
    plot_optimization = True
    """Whether to show an animation of the string selection process during the optimization."""
    name_of_the_run = 'test'
    use_cpus = 1
    """Number of cpus to use for parallelization. Set to 0 to use all available cpus."""

    @property
    def pin_width(self) -> float:
        """physical width of a pin in pixel"""
        return self.pin_side_length / self.string_thickness

    @property
    def high_res(self) -> int:
        """resolution/width of the frame in pixel. should be a power of 2. Set frame_diameter and string_thickness accordingly."""
        return int(self.frame_diameter // self.string_thickness)

    @property
    def low_res(self) -> int:
        return int(self.high_res // self.super_sampling_window_width)


__config = None


def get_config() -> Config:
    global __config
    if __config is None:
        __config = Config()
    return __config
