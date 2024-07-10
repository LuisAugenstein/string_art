import torch
import matplotlib.pyplot as plt
from string_art import pins
import string_art.edges as edges
from string_art.string_matrix import get_string_matrix, line_profile
from string_art.optimization import naive_greedy, StringPath
from string_art.image import load_input_image, create_circular_mask

IMAGE_SIZE = 256
STRING_WIDTH = 0.5
STRING_COLOR = 0.8
N_PINS = 90
TARGET_IMAGE_PATH = 'data/inputs/cat.png'

torch.set_default_dtype(torch.float64)


img = load_input_image(TARGET_IMAGE_PATH, IMAGE_SIZE)  # [IMAGE_SIZE, IMAGE_SIZE]
circular_mask = create_circular_mask(IMAGE_SIZE, radius=IMAGE_SIZE//2)  # [IMAGE_SIZE, IMAGE_SIZE]
img[~circular_mask] = 0.
pin_positions = pins.point_based(N_PINS, IMAGE_SIZE)  # [N_pins, 2] first pin on the right then moving counter-clockwise

# plot original image and empty reconstruction
fig, [ax_reconstruction, ax_img] = plt.subplots(1, 2)  # Create a figure and axis
ax_img.imshow(img, cmap='gray')
ax_reconstruction.scatter(pin_positions[:, 1].numpy(), pin_positions[:, 0].numpy(), c='black', s=5)
reconstruction_plot = ax_reconstruction.imshow(torch.ones(IMAGE_SIZE, IMAGE_SIZE), cmap='gray', vmin=0, vmax=1)

edges_index_based = edges.index_based(N_PINS)  # [N_strings, 2]
edges_pin_based = edges.point_based(pin_positions, edges_index_based)  # [N_pins, 2, 2]
string_matrix = get_string_matrix(edges_pin_based, line_profile.trapez(STRING_WIDTH, STRING_COLOR), IMAGE_SIZE)  # [IMAGE_SIZE**2, N_strings]
string_matrix[~circular_mask.flatten(), :] = 0.
img_vector = img.reshape(-1)  # [HW]
path = StringPath(edges_index_based, start_pin_index=torch.zeros(1))
string_img = torch.zeros(IMAGE_SIZE, IMAGE_SIZE)
best_loss = torch.inf
for step in range(300):
    edge_index, loss = naive_greedy(string_matrix, img_vector, path)
    if loss >= best_loss:
        print("Optimization done")
        break
    best_loss = loss
    print(f"{step:4d} {loss.item():.10f} {edge_index.item():4d}")
    path.add_edge_index(edge_index)

    new_string = string_matrix[:, path.edge_path[-1]]  # [HW]
    string_img += new_string.reshape(IMAGE_SIZE, IMAGE_SIZE)

    ax_reconstruction.set_title(f'image reconstruction step={step}')  # Update the title
    reconstruction_plot.set_data(string_img)
    plt.draw()
    plt.pause(0.01)

plt.show()
