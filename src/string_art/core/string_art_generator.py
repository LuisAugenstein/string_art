import time
import torch
from torchvision.transforms.v2 import Transform, Compose, ToDtype, Resize, Grayscale, RandomInvert
from typing import Protocol
from dataclasses import dataclass
from torchvision.io import decode_image
from string_art.core.string_art_config import StringArtConfig
from string_art.core.string_art_reconstruction import StringArtReconstruction
from string_art.core.string_art_store import StringArtStore
from string_art.algorithms import StringArtAlgorithm
from string_art.algorithms.naive import NaiveAlgorithmConfig, NaiveAlgorithm
from string_art.algorithms.radon import RadonAlgorithmConfig, RadonAlgorithm
from string_art.visualization import StringArtVisualizer
from string_art.visualization.default_visualizer import DefaultVisualizer

@dataclass
class StringArtGeneratorConfig(Protocol):
    image_width: int
    """resolution of the quadratic input image in pixels."""

class StringArtGenerator:
    store: StringArtStore
    algorithm: StringArtAlgorithm
    config: StringArtGeneratorConfig
   
    def __init__(self, config: StringArtConfig):
        self.config = config
        self.store = StringArtStore(config)
        self.algorithm = self._get_algorithm(config, self.store)

    def load_image(self, image_path: str, 
                   transforms: list[Transform] | None = None) -> torch.Tensor:
        """Load an image and apply preprocessing. The standard preprocessing steps are resizing, converting to grayscale 
        and inverting the image since the strings are typically black on a light background.
        The meaning of the output image values is: 1="a string should span over this pixel" and 0="no string should span over this pixel. The light background should fully shine through"
        
        Parameters
        -
        image_path: Path to the image file
        transforms: List of torchvision transforms to apply after loading the image.
                    If None, default preprocessing for string art is applied
            
        Returns:
        -
        torch.Tensor: [1, H, W] Preprocessed grayscale image with values between 0 (white) and 1 (black) 
        """
        img = decode_image(image_path, mode='RGB')
        transform = Compose(transforms if transforms != None else  [
            Resize(size=(self.config.image_width, self.config.image_width)),
            Grayscale(),
            RandomInvert(p=1.0),
            ToDtype(torch.float32, scale=True)
        ])
        return transform(img)
    
    def add_visualizer(self, visualizer: StringArtVisualizer | None = None) -> None:
        self.store.register(DefaultVisualizer(self.config, self) if visualizer is None else visualizer)

    def generate(self, image: torch.Tensor) -> StringArtReconstruction:
        self.store.image = image
        precomputed_reconstruction = self.store.load()
        if precomputed_reconstruction is not None:
            return precomputed_reconstruction

        print('Start generating string art reconstruction')
        start = time.time()
        reconstruction = self.algorithm.generate()
        print(f'Completed after {time.time()-start} seconds')

        self.store.save()
        return reconstruction

    def _get_algorithm(self, config: StringArtConfig, store: StringArtStore) -> StringArtAlgorithm:
        match config:
            case NaiveAlgorithmConfig():
                return NaiveAlgorithm(config, store)
            case RadonAlgorithmConfig():
                return RadonAlgorithm(config, store)
            case _:
                raise ValueError(f"Unknown algorithm config: {config}")

