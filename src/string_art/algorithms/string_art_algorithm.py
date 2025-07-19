
import torch
from abc import ABC, abstractmethod
from string_art.core.string_art_store import StringArtStore
from string_art.core.string_art_config import StringArtConfig
from string_art.core.string_art_reconstruction import StringArtReconstruction

class StringArtAlgorithm(ABC):
    config: StringArtConfig
    store: StringArtStore

    def __init__(self, config: StringArtConfig, store: StringArtStore):
        self.config = config
        self.store = store

    @abstractmethod
    def generate(image: torch.Tensor) -> StringArtReconstruction:
        ...