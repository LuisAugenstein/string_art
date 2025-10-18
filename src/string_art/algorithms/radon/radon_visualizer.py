from matplotlib.axes import Axes
from matplotlib import pyplot as plt
from string_art.algorithms.radon.radon_algorithm_config import RadonAlgorithmConfig
from string_art.algorithms.radon.radon_reconstruction import RadonReconstruction
from string_art.core.default_visualizer import DefaultVisualizer
from string_art.core.string_art_store import StringArtStore
import numpy as np

class RadonVisualizer(DefaultVisualizer):

    config: RadonAlgorithmConfig

    def __init__(self, config: RadonAlgorithmConfig, store: StringArtStore):
        super().__init__(config, store)
    
    def create_figure(self) -> None:
        self.fig, [ax_img, ax_reconstruction, ax_radon] = plt.subplots(1, 3, figsize=(15, 7))
        self._plot_img(ax_img, self.store.image.squeeze().numpy())
        self.reconstruction_title, self.line_collection = self._plot_reconstruction(ax_reconstruction)
        self._plot_radon(ax_radon)

    def _plot_radon(self, ax_radon: Axes) -> None:
        ax_radon.set_ylabel(f's [{self.config.image_width}] - line distance from center')
        ax_radon.set_xlabel(r'$\alpha$' + f' [{self.config.n_radon_angles}]- line angle')
        ax_radon.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
        ax_radon.set_xticklabels(['0', r'$\frac{1}{4}\pi$', r'$\frac{1}{2}\pi$', r'$\frac{3}{4}\pi$', r'$\pi$'])
        ax_radon.set_yticks([1, 0.5, 0, -0.5, -1])
        reconstruction: RadonReconstruction = self.store.reconstruction
        ax_radon.imshow(reconstruction.initial_radon_image, cmap='hot', extent=(0, np.pi, 1, -1), aspect=np.pi/2)