import torch
from typing import Protocol
from string_art.core.string_art_reconstruction import StringArtReconstruction


class StringArtListener(Protocol):
    def notify(self, image: torch.Tensor, reconstruction: StringArtReconstruction) -> None:
        ...