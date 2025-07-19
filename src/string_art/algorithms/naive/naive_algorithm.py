import torch
from string_art.core.string_art_store import StringArtStore
from string_art.algorithms.string_art_algorithm import StringArtAlgorithm
from string_art.core.string_art_reconstruction import StringArtReconstruction
from string_art.algorithms.naive.naive_algorithm_config import NaiveAlgorithmConfig

class NaiveAlgorithm(StringArtAlgorithm):
    config: NaiveAlgorithmConfig

    def __init__(self, config: NaiveAlgorithmConfig, store: StringArtStore):
        super().__init__(config, store)

    def generate(self, image: torch.Tensor) -> StringArtReconstruction:
        return StringArtReconstruction()