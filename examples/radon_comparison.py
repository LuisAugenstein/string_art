import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import string_art.pins as pins
from string_art.string_reconstruction_radon import string_reconstruction_radon, StringReconstructionRadonConfig, StringReconstructionRadonCallbackConfig

torch.set_default_dtype(torch.float64)

configs = [
    StringReconstructionRadonConfig(n_pins=100, n_radon_angles=100),
    StringReconstructionRadonConfig(n_pins=100, n_radon_angles=200),
    StringReconstructionRadonConfig(n_pins=100, n_radon_angles=300),
    StringReconstructionRadonConfig(n_pins=200, n_radon_angles=100),
    StringReconstructionRadonConfig(n_pins=200, n_radon_angles=200),
    StringReconstructionRadonConfig(n_pins=200, n_radon_angles=300),
    StringReconstructionRadonConfig(n_pins=300, n_radon_angles=100),
    StringReconstructionRadonConfig(n_pins=300, n_radon_angles=200),
    StringReconstructionRadonConfig(n_pins=300, n_radon_angles=300),
]

IMAGE_SIZE = 400
TARGET_IMAGE_PATH = f'data/inputs/cat_400.png'
LINE_TRANSPARENCY = 0.06

# Convert Image to black and white
img = Image.open(TARGET_IMAGE_PATH)
img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
img = np.array(img)
if len(img.shape) == 3:
    img = img.mean(axis=-1)
img = 1 - img/255


class PlotFinalResultCallback:
    line_segments = []

    def __call__(self, config: StringReconstructionRadonCallbackConfig) -> None:
        s_index, alpha_index = config.reconstructed_line_radon_index_based
        print(config.step+1, s_index, alpha_index, config.residual)

        s, alpha = config.reconstructed_line_radon_parameter_based
        psi_1, psi_2 = alpha - np.arccos(s), alpha + np.arccos(s)
        start_point = [np.cos(psi_1), np.sin(psi_1)]
        end_point = [np.cos(psi_2), np.sin(psi_2)]
        self.line_segments.append([start_point, end_point])


fig, axs = plt.subplots(3, 3, figsize=(15, 7), sharex=True, sharey=True)

for i, config in enumerate(configs):
    callback = PlotFinalResultCallback()
    config.n_max_steps = 5000
    string_reconstruction_radon(img, config, callback)
    line_collection = LineCollection(callback.line_segments, colors=[(0, 0, 0, LINE_TRANSPARENCY)])
    ax = axs[i // axs.shape[0], i % axs.shape[1]]
    pins_point_based = pins.point_based(config.n_pins)
    ax.scatter(pins_point_based[:, 0], pins_point_based[:, 1], c=[[0.8, 0.5, 0.2]], s=5)
    ax.set_aspect('equal')

    ax.add_collection(line_collection)
    ax.set_title(f'{config.n_pins} pins, {config.n_radon_angles} angles')

fig.tight_layout()

plt.show()
