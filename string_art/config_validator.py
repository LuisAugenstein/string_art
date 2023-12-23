from math import log2


def is_resolution_power_of_2(thread_thickness: float, frame_diameter: float) -> bool:
    log_resolution = log2(frame_diameter / thread_thickness)
    is_power_of_2 = int(log_resolution) == log_resolution
    return is_power_of_2
