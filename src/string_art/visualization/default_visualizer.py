
from typing import Text
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from string_art import pins
from string_art.core import StringArtConfig, StringArtStore
from string_art.visualization import StringArtVisualizer
import matplotlib.pyplot as plt
import numpy as np
from string_art import edges
from matplotlib import animation

class DefaultVisualizer(StringArtVisualizer):

    fig: Figure
    reconstruction_title: Text
    line_collection: LineCollection

    def __init__(self, config: StringArtConfig, store: StringArtStore):
        super().__init__(config, store)

    def create_figure(self) -> None:
        self.fig, [ax_img, ax_reconstruction] = plt.subplots(1, 2, figsize=(15, 7))
        self._plot_img(ax_img, self.store.image.squeeze().numpy())
        self.reconstruction_title, self.line_collection = self._plot_reconstruction(ax_reconstruction)

    def _plot_img(self, ax_img: Axes, image: np.ndarray) -> None:
        ax_img.set_title('Original Image')
        ax_img.set_xlabel('column')
        ax_img.set_ylabel('row')
        ax_img.imshow(1-image, cmap='gray', vmin=0, vmax=1, extent=(0, self.config.image_width, self.config.image_width, 0))

    def _plot_reconstruction(self, ax_reconstruction: Axes) -> tuple[Text, LineCollection]:
        reconstruction_title = ax_reconstruction.set_title(f'Reconstruction - {self.config.n_pins} pins, {0} strings')
        ax_reconstruction.set_xlabel('x')
        ax_reconstruction.set_ylabel('y')
        ax_reconstruction.set_xticks([1, 0.5, 0, -0.5, -1])
        ax_reconstruction.set_yticks([1, 0.5, 0, -0.5, -1])
        pins_point_based = pins.point_based(self.config.n_pins)
        ax_reconstruction.scatter(pins_point_based[:, 0], pins_point_based[:, 1], c=[[0.8, 0.5, 0.2]], s=5)
        ax_reconstruction.set_aspect('equal')
        line_collection = LineCollection([], colors=[(0, 0, 0, self.config.visualizer.line_transparency)])
        ax_reconstruction.add_collection(line_collection)
        return reconstruction_title, line_collection
    
    def update(self) -> None:
        self._animate(self.store.reconstruction.strings.shape[0])

    def show_animation(self, save_as_mp4=False) -> None:
        """starts an animation using the stored StringArtReconstruction"""
        self.create_figure()
        n_strings = self.store.reconstruction.strings.shape[0]
        if save_as_mp4:
            ani = animation.FuncAnimation(fig=self.fig, func=self._animate, frames=n_strings, interval=10, repeat=False)
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=250, bitrate=1800)
            ani.save(f'animation.mp4', writer=writer, progress_callback=lambda i, n: print(f'{i}/{n}'))
        else:
            for i in range(0, n_strings+1, 50):
                self._animate(i)
                plt.draw()
                plt.pause(0.0001)
            self._animate(n_strings)
            plt.show()

    def _animate(self, i: int) -> tuple[LineCollection, Text]:
        if i == 0:
            return self.line_collection, self.reconstruction_title
        
        segments_visualized = self.line_collection.get_segments()
        n_strings_visualized = len(segments_visualized)
        strings_to_draw_angle_based = self.store.reconstruction.strings[n_strings_visualized:i]
        strings_to_draw_point_based = edges.angle_to_point_based(strings_to_draw_angle_based)
        segments_visualized.extend(strings_to_draw_point_based.numpy())
        self.line_collection.set_segments(segments_visualized)
        n_strings_visualized += strings_to_draw_point_based.shape[0]
        self.reconstruction_title.set_text(f'Reconstruction - {self.config.n_pins} pins, {n_strings_visualized} strings')
        return self.line_collection, self.reconstruction_title
    
      

