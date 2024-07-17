import torch
import matplotlib.pyplot as plt
from skimage.transform import radon
import string_art.pins as pins
import string_art.edges as edges
import string_art.line_profile as line_profile
from string_art.image import create_circular_mask, load_input_image

torch.set_default_dtype(torch.float64)

IMAGE_SIZE = 512
N_RADON_ANGLES = 512  # typically set to same as IMAGE_SIZE
STRING_WIDTH = 0.25
STRING_COLOR = 1
N_PINS = 512
TARGET_IMAGE_PATH = 'data/inputs/cat_400.png'
N_STEPS = 10000

# setup pins and edges in the different necessary representations
pins_angle_based = pins.angle_based(N_PINS)  # [N_pins]
pins_point_based = pins.point_based(pins_angle_based, IMAGE_SIZE)  # [N_pins, 2]
edges_index_based = edges.index_based(N_PINS)  # [N_strings, 2]
edges_point_based = edges.point_based(pins_point_based, edges_index_based)  # [N_pins, 2, 2]
edges_angle_based = edges.angle_based(pins_angle_based, edges_index_based)  # [N_strings, 2]
radon_angles_radians = torch.arange(N_RADON_ANGLES) * torch.pi / N_RADON_ANGLES  # [N_RADON_ANGLES]
radon_angles_degrees = torch.arange(N_RADON_ANGLES) * 180 / N_RADON_ANGLES
s_indices, alpha_indices = edges.radon_index_based(edges_angle_based, radon_angles_radians, IMAGE_SIZE).T  # [N_strings] [N_strings]
valid_radon_parameters_mask = torch.zeros(IMAGE_SIZE, N_RADON_ANGLES, dtype=torch.bool)
valid_radon_parameters_mask[s_indices, alpha_indices] = True
radon_beam_lengths = (torch.sqrt(1 - (1 - torch.linspace(0, 2, IMAGE_SIZE)) ** 2) *
                      IMAGE_SIZE).unsqueeze(1).repeat(1, N_RADON_ANGLES)  # [IMAGE_SIZE, N_RADON_ANGLES]
radon_beam_lengths[radon_beam_lengths == 0] = torch.inf

# load img and compute radon transform
circular_mask = create_circular_mask(IMAGE_SIZE, radius=IMAGE_SIZE//2 - 5)  # [IMAGE_SIZE, IMAGE_SIZE]
img = load_input_image(TARGET_IMAGE_PATH, IMAGE_SIZE)  # [IMAGE_SIZE, IMAGE_SIZE]
if len(img.shape) == 3:
    img = img.mean(dim=-1)
img = 1 - img
img[~circular_mask] = 0.
img_radon = torch.tensor(radon(img, radon_angles_degrees))  # [IMAGE_SIZE, N_RADON_ANGLES]
img_radon /= radon_beam_lengths


# plot original image and empty reconstruction
fig, [ax_img, ax_reconstruction, ax_radon] = plt.subplots(1, 3)  # Create a figure and axis
ax_img.imshow(1-img, cmap='gray',  vmin=0, vmax=1, extent=(0, IMAGE_SIZE, IMAGE_SIZE, 0))
reconstruction_plot = ax_reconstruction.imshow(torch.zeros(IMAGE_SIZE, IMAGE_SIZE), cmap='gray', vmin=0, vmax=1)
ax_reconstruction.scatter(pins_point_based[:, 1].numpy(), pins_point_based[:, 0].numpy(), c=[[0.8, 0.5, 0.2]], s=5)
available_lines_img = torch.zeros(IMAGE_SIZE, IMAGE_SIZE)
# img_radon[~valid_radon_parameters_mask] = 0
radon_plot = ax_radon.imshow(img_radon, cmap='hot', extent=(0, 180, IMAGE_SIZE, 0), aspect=180/IMAGE_SIZE)

# optimize string image
string_img = torch.zeros(IMAGE_SIZE, IMAGE_SIZE)
residual_threshold = 1e-5
residual = img_radon.clone()
solution_path = []
for step in range(300):
    plt.pause(0.2)
    # argmax takes 10% of the time. Maybe speed this up by only considering the valid strings
    s_index, alpha_index = torch.unravel_index(torch.argmax(residual), residual.shape)
    if residual[s_index, alpha_index] < residual_threshold:
        print(f"{step:4d} {residual[s_index, alpha_index].item():.10f}")
        print(f"Optimization done")
        break
    edge_index = torch.where((s_indices == s_index) & (alpha_indices == alpha_index))[0].squeeze()  # []
    print(f"{step:4d} {residual[s_index, alpha_index].item():.10f} {edge_index.item():4d}")
    edge_image = edges.get_image(edges_point_based[edge_index], line_profile.trapez(STRING_WIDTH, STRING_COLOR), IMAGE_SIZE)
    edge_image[~circular_mask] = 0.

    edge_radon = radon(edge_image, radon_angles_degrees)  # this line takes 80% of the time. Lets do this analytically
    edge_radon /= radon_beam_lengths
    residual = torch.clamp(residual - edge_radon, 0, torch.inf)
    residual[s_index, alpha_index] = 0
    solution_path.append(edge_index)

    string_img += edge_image
    ax_reconstruction.set_title(f'image reconstruction step={step}')  # Update the title
    reconstruction_plot.set_data(1-string_img)
    # TODO: Maybe normalize the residual before plotting or adjust vmin and vmax of the plot each time we update the data.
    radon_plot.set_data(residual)
    plt.draw()

plt.show()
