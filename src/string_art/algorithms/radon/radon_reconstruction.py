import torch
from dataclasses import dataclass
from string_art.core.string_art_reconstruction import StringArtReconstruction
import string_art.edges as edges

@dataclass
class RadonReconstruction(StringArtReconstruction):
    initial_radon_image: torch.Tensor | None = None

    def add_string(self, string_radon_parameter_based: tuple[float, float]) -> None:
        string_angle_based = edges.radon_parameter_to_angle_based(torch.tensor(string_radon_parameter_based).unsqueeze(0))
        if self.strings is None:
            self.strings = string_angle_based
        else:
            self.strings = torch.cat([self.strings, string_angle_based])