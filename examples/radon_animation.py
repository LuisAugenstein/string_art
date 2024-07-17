import time
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection
from string_art.string_reconstruction_radon import string_reconstruction_radon, StringReconstructionRadonCallbackConfig, StringReconstructionRadonConfig
import string_art.pins as pins

torch.set_default_dtype(torch.float64)

IMAGE_SIZE = 400
TARGET_IMAGE_PATH = f'data/inputs/cat_400.png'
LINE_TRANSPARENCY = 0.06
RENDER_VIDEO = False
config = StringReconstructionRadonConfig(
    n_max_steps=5000,
    n_pins=150,
    n_radon_angles=300
)

# Convert Image to black and white
img = Image.open(TARGET_IMAGE_PATH)
img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
img = np.array(img)
if len(img.shape) == 3:
    img = img.mean(axis=-1)
img = 1 - img/255


class MemoryCallback:
    line_segments = []
    radon_imgs = []

    def __call__(self, config: StringReconstructionRadonCallbackConfig) -> None:
        s_index, alpha_index = config.reconstructed_line_radon_index_based
        if (config.step+1) % 10 == 0:
            print(config.step+1, s_index, alpha_index, config.residual)

        s, alpha = config.reconstructed_line_radon_parameter_based
        psi_1, psi_2 = alpha - np.arccos(s), alpha + np.arccos(s)
        start_point = [np.cos(psi_1), np.sin(psi_1)]
        end_point = [np.cos(psi_2), np.sin(psi_2)]
        self.line_segments.append([start_point, end_point])
        self.radon_imgs.append(config.img_radon)


callback = MemoryCallback()
start = time.time()
string_reconstruction_radon(img, config, callback)
print(f'Completed after {time.time()-start} seconds')

fig, [ax_img, ax_reconstruction, ax_radon] = plt.subplots(1, 3, figsize=(15, 7))  # Create a figure and axis
ax_img.set_title('Original Image')
ax_img.set_xlabel('column')
ax_img.set_ylabel('row')
reconstruction_title = ax_reconstruction.set_title(f'Reconstruction - {config.n_pins} pins, {0} strings')
ax_reconstruction.set_xlabel('x')
ax_reconstruction.set_ylabel('y')
ax_reconstruction.set_xticks([1, 0.5, 0, -0.5, -1])
ax_reconstruction.set_yticks([1, 0.5, 0, -0.5, -1])
radon_title = ax_radon.set_title(f'Radon Transformed Residual - {0} strings')
ax_radon.set_ylabel(f's [{IMAGE_SIZE}] - line distance from center')
ax_radon.set_xlabel(r'$\alpha$' + f' [{config.n_radon_angles}]- line angle')
pi_positions = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
ax_radon.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
ax_radon.set_xticklabels(['0', r'$\frac{1}{4}\pi$', r'$\frac{1}{2}\pi$', r'$\frac{3}{4}\pi$', r'$\pi$'])
ax_radon.set_yticks([1, 0.5, 0, -0.5, -1])

ax_img.imshow(1-img, cmap='gray',  vmin=0, vmax=1, extent=(0, IMAGE_SIZE, IMAGE_SIZE, 0))
pins_point_based = pins.point_based(config.n_pins)
ax_reconstruction.scatter(pins_point_based[:, 0], pins_point_based[:, 1], c=[[0.8, 0.5, 0.2]], s=5)
ax_reconstruction.set_aspect('equal')
line_collection = LineCollection([], colors=[(0, 0, 0, LINE_TRANSPARENCY)])
ax_reconstruction.add_collection(line_collection)
radon_plot = ax_radon.imshow(callback.radon_imgs[0], cmap='hot', extent=(0, np.pi, 1, -1), aspect=np.pi/2)


def animate(i: int):
    if i == 0:
        return line_collection, radon_plot, reconstruction_title, radon_title

    line_collection.set_segments([*line_collection.get_segments(), callback.line_segments[i-1]])
    reconstruction_title.set_text(f'Reconstruction - {config.n_pins} pins, {i+1} strings')
    radon_title.set_text(f'Radon Transformed Residual - {i+1} strings')
    radon_plot.set_array(callback.radon_imgs[i])
    return line_collection, radon_plot, reconstruction_title, radon_title


if RENDER_VIDEO:
    ani = animation.FuncAnimation(fig=fig, func=animate, frames=len(callback.line_segments), interval=10, repeat=False)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=250, bitrate=1800)
    ani.save(f'cat_{IMAGE_SIZE}_{config.n_pins}_{config.n_radon_angles}.mp4', writer=writer, progress_callback=lambda i, n: print(f'{i}/{n}'))
else:
    for i in range(config.n_max_steps):
        animate(i)
        if (i+1) % 10 == 0:
            plt.draw()
            plt.pause(0.0001)
    plt.show()
