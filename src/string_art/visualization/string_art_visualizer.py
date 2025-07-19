import torch
from string_art.core.string_art_config import StringArtConfig
from string_art.core.string_art_reconstruction import StringArtReconstruction


class StringArtVisualizer:
    config: StringArtConfig

    def __init__(self, config: StringArtConfig):
        self.config = config

    def notify(self, image: torch.Tensor, reconstruction: StringArtReconstruction) -> None:
        ...