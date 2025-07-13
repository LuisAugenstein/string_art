import torch
from string_art.line_profile import LineProfile


def index_based(N_pins: int) -> torch.Tensor:
    """
    N_strings = N_pins * (N_pins - 1) / 2 = N_pins choose 2

    Parameters
    -
    N_pins: int        total number of pins

    Returns
    -
    edges_index_based: [N_strings, 2]  all possible edges between any two pins. The first entry is always the lower pin index, i.e., pin_index_lower, pin_index_higher = edge_indices[i]
    """
    pin_indices = torch.arange(N_pins)
    string_indices = torch.combinations(pin_indices, r=2)  # [N_strings, 2]
    return string_indices


def point_based(points: torch.Tensor, edges_index_based: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    -
    positions: [N_pins, 2]             N_pins two-dimensional points
    edges_index_based: [N_strings, 2]  N_strings edges each defined by ids that can be interpreted as indices of the points tensor, e.g., point_index, point_index2 = edge_indices[i]

    Returns
    -
    edges_point_based: [N_strings, 2, 2] N_strings edges each defined by two points p1, p2 = edges[i]
    """
    return points[edges_index_based]  # [N_strings, 2, 2]


def angle_based(pins_angle_based: torch.Tensor, edges_index_based: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    -
    pin_angles: [N_pins]               N_pins angles of the pins
    edges_index_based: [N_strings, 2]  N_strings edges each defined by ids that can be interpreted as indices of the points tensor, e.g., point_index, point_index2 = edge_indices[i]

    Returns
    -
    edges_angle_based: [N_strings, 2] N_strings edges each defined by two angles psi1, psi2 = edges[i]
    """
    return pins_angle_based[edges_index_based]  # [N_strings, 2]


def radon_parameter_based(edges_angle_based: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    -
    edges_angle_based: [N_strings, 2]  N edges each defined by two angles psi1, psi2 = edges[i]

    Returns
    -
    edges_radon_parameter_based: [N_strings, 2]  N edges each defined by their two radon parameters s, alpha = edges[i]
    """
    def _mean_angle(psi_0: torch.Tensor, psi_1: torch.Tensor) -> torch.Tensor:
        """Computes the mean angle between psi_0 and psi_1 in the range [0, 2*pi)"""
        mean_angle = (psi_0 + psi_1) / 2  # [N]
        mean_angle[(psi_1 - psi_0) > torch.pi] -= torch.pi  # [N]
        return mean_angle % (2*torch.pi)  # [N]

    psi_0, psi_1 = edges_angle_based[:, 0], edges_angle_based[:, 1]  # [N_strings], [N_strings]
    mean_angle = _mean_angle(psi_0, psi_1)  # [N_strings]
    s = torch.abs(torch.exp(edges_angle_based * 1j).mean(dim=1))  # [N]
    s[mean_angle >= torch.pi] *= -1  # [N]
    alpha = mean_angle % torch.pi  # [N]
    return torch.stack([s, alpha], dim=1)


def radon_index_based(edges_radon_parameter_based: torch.Tensor, s_domain, alpha_domain: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    -
    edges_radon_parameter_based: [N_strings, 2]  N edges each defined their two radon parameters s, alpha = edges[i]
    s_domain: [N_s]                    possible values for the radon parameter s (distance to center) in [-1, 1]
    alpha_domain: [N_alpha]            possible values for the radon parameter alpha (angle) in [0, pi)

    Returns
    -
    edges_radon_index_based: [N_strings, 2] N edges each defined by the domain indices of their two radon parameters s_index, alpha_index = edges[i]
                             s_index from 0 to N_s-1, alpha_index from 0 to N_alpha-1
    """
    def _find_nearest_domain_index(values: torch.Tensor, domain: torch.Tensor) -> torch.Tensor:
        diffs = torch.abs(values.unsqueeze(1) - domain.unsqueeze(0))  # [N_values, N_domain]
        return torch.argmin(diffs, dim=1)  # [N_values]

    s, alpha = edges_radon_parameter_based.T  # [N_strings], [N_strings]
    s_indices = _find_nearest_domain_index(s, s_domain)  # [N_strings]
    alpha_indices = _find_nearest_domain_index(alpha, alpha_domain)  # [N_strings]
    return torch.stack([s_indices, alpha_indices], dim=1)  # [N_strings, 2]


def distance_points_edge(points: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    -
    points: [N, 2]  N two-dimensional points
    edge: [2, 2]    edge defined by two points p_start, p_end = edges[i]

    Returns
    distances: [N] distance of each point to the edge/line
    """
    p_start, p_end = edge[0], edge[1]  # [2], [2]
    rot_90 = torch.tensor([[0., -1.], [1., 0.]])  # [2, 2]
    normal = rot_90 @ (p_end - p_start)  # [2]
    u = normal / torch.norm(normal)  # [2]
    h = u.unsqueeze(0) @ p_start  # [1]
    distances = torch.abs((points @ u).squeeze() - h)  # [N]
    return distances


def get_image(edge: torch.Tensor, line_profile: LineProfile, image_size: int) -> torch.Tensor:
    """
    Parameters
    -
    edges: [2, 2]  edge defined by two points p_start, p_end = edges[i] that represent points on the unit circle
    line_profile:  function determininig the pixel intensity based on its distance to the edge
    image_size:    width/height of the square target image

    Returns
    -
    edge_image: [image_size, image_size] image of the edge. Most pixels are 0.
    """
    row, col = torch.meshgrid(torch.linspace(-1, 1, image_size), torch.linspace(-1, 1, image_size))  # [image_size, image_size]
    grid = torch.stack([col, -row], dim=-1).reshape(-1, 2).to(edge.dtype)  # [image_size**2, 2]
    distances = distance_points_edge(grid, edge)  # [image_size**2]
    edge_image = line_profile(distances).reshape(image_size, image_size)  # [image_size, image_size]
    return edge_image
