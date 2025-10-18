import torch
from string_art import edges, pins
from string_art.algorithms.radon.radon_reconstruction import RadonReconstruction
from string_art.core.string_art_store import StringArtStore
from string_art.algorithms.string_art_algorithm import StringArtAlgorithm
from string_art.algorithms.radon.radon_algorithm_config import RadonAlgorithmConfig
from string_art.utils import create_circular_mask
import numpy as np
from skimage.transform import radon

class RadonAlgorithm(StringArtAlgorithm):
    config: RadonAlgorithmConfig

    def __init__(self, config: RadonAlgorithmConfig, store: StringArtStore):
        super().__init__(config, store)

    def generate(self) -> RadonReconstruction:
        image = self.store.image.squeeze()
        image[create_circular_mask(self.config.image_width)] = 0

        # setup pins and edges in the different necessary representations
        pins_angle_based = pins.angle_based(self.config.n_pins)  # [N_pins]
        edges_index_based = edges.index_based(self.config.n_pins)  # [N_edges, 2]

        if self.config.minimum_pin_span > 1:
            distance = torch.abs(edges_index_based[:, 1] - edges_index_based[:, 0])
            pin_span = torch.min(distance, self.config.n_pins - distance)
            edges_index_based = edges_index_based[pin_span >= self.config.minimum_pin_span]

        edges_angle_based = edges.angle_based(pins_angle_based, edges_index_based)  # [N_edges, 2]
        alpha_domain = torch.linspace(0, torch.pi, self.config.n_radon_angles) # [N_alpha=N_radon_angles]
        s_domain = torch.linspace(-1, 1, self.config.image_width) # [N_s=image_width]
        edges_radon_parameter_based = edges.angle_to_radon_parameter_based(edges_angle_based)  # [N_edges, 2]
        s_indices, alpha_indices = edges.radon_parameter_to_radon_index_based(edges_radon_parameter_based, s_domain, alpha_domain).T  # [N_edges] [N_edges]
        valid_radon_parameters_mask = torch.zeros(self.config.image_width, self.config.n_radon_angles, dtype=torch.bool)
        valid_radon_parameters_mask[s_indices, alpha_indices] = True

        alpha_domain_deg = np.linspace(0, 180, self.config.n_radon_angles)
        img_radon = radon(image.numpy(), alpha_domain_deg)
        img_radon = img_radon / self.config.image_width

        s_domain = s_domain.numpy()
        alpha_domain = alpha_domain.numpy()

        _, S = np.meshgrid(alpha_domain, s_domain)  # [N_RADON_ANGLES, self.config.image_width]
        line_lengths = 2 * np.sqrt(1 - S ** 2)
        zero_length_mask = line_lengths == 0
        # we are intereseted in the radon value per unit length
        img_radon[~zero_length_mask] = img_radon[~zero_length_mask] / line_lengths[~zero_length_mask]
        img_radon[zero_length_mask] = 0
        # remove all lines that cannot be spanned between two pins
        img_radon[~valid_radon_parameters_mask] = 0

        reconstruction = RadonReconstruction(initial_radon_image=img_radon)

        for step in range(self.config.n_strings):
            s_index, alpha_index = np.unravel_index(np.argmax(img_radon), img_radon.shape)
            residual = img_radon[s_index, alpha_index]
            if residual < self.config.residual_threshold:
                print('Optimization finished')
                return

            alpha_indomain, s_indomain = alpha_domain[alpha_index], s_domain[s_index]
            p_line_theory = self._analytical_radon_line(alpha_indomain, s_indomain, alpha_domain, s_domain, line_lengths, self.config.t_start,
                                                self.config.t_end, self.config.line_darkness, self.config.p_min)
            img_radon = img_radon - p_line_theory
            img_radon[s_index, alpha_index] = 0

            if (step+1) % 10 == 0:
                print(f"{step+1:5d} {s_index:3d} {alpha_index:3d} {residual:5.4f}")
            closest = torch.argmin((edges_radon_parameter_based - torch.tensor([s_indomain, alpha_indomain])).norm(dim=1))
            s, alpha = edges_radon_parameter_based[closest]
            reconstruction.add_string(string_radon_parameter_based=(s, alpha))
            self.store.update(reconstruction)
            
        return reconstruction

    def _analytical_radon_line(self, alpha: float, s: float, alpha_domain, s_domain, line_lengths, t_start, t_end, line_darkness, p_min):
        ALPHA, S = np.meshgrid(alpha_domain, s_domain)
        ALPHA_diff_alpha0 = ALPHA - alpha
        sin_ALPHA_alpha0 = np.sin(ALPHA_diff_alpha0)
        sin_ALPHA_alpha0_squared = sin_ALPHA_alpha0 ** 2

        n = 4
        t = np.linspace(t_start, t_end, n)
        nominator = (S ** 2 + s ** 2 - 2 * S * s * np.cos(ALPHA_diff_alpha0))
        p_regionfuns = [nominator / (sin_ALPHA_alpha0_squared + t_i) - 1 for t_i in t]

        mask = np.zeros_like(ALPHA)
        mask[p_regionfuns[0] < 0] = 1
        for i in range(n - 1):
            mask[(p_regionfuns[i] > 0) & (p_regionfuns[i+1] < 0)] = (t_end - t[i]) / (t_end - t_start)

        p_line = line_darkness * p_min / ((line_darkness * line_lengths - p_min) * np.abs(sin_ALPHA_alpha0) + p_min)
        return p_line * mask