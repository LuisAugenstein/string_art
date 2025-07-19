from abc import ABC
from dataclasses import dataclass, fields
from string_art.units import Length

def pretty_print(cls):
    def __str__(self):
        field_strings = []
        for field_ in fields(self):
            value = getattr(self, field_.name)
            if isinstance(value, str):
                field_strings.append(f"    {field_.name}='{value}'")
            else:
                field_strings.append(f"    {field_.name}={value}")
        return f"{self.__class__.__name__}(\n" + ",\n".join(field_strings) + "\n)"
    cls.__str__ = __str__
    return cls

@pretty_print
@dataclass
class StringArtConfig(ABC):
    n_pins: int = 300  
    """Number of pins around the circular canvas."""

    image_width: int = 400
    """resolution of the quadratic input image in pixels."""
    
    n_strings: int = 6000  
    """maximum number of strings to use for the reconstruction."""

    pin_radius: Length = Length.mm(3)
    """physical radius of a circular pin."""

    string_thickness: Length = Length.mm(1)
    """physical thickness of a string"""
    
    canvas_diameter: Length = Length.cm(30)
    """Physical diameter of the circular canvas"""

    store_path: str = "data/stores"
    """Path to the output directory where the reconstructions and config is stored"""