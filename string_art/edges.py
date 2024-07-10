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
    positions: [N, 2]          N two-dimensional points
    edges_index_based: [M, 2]  M edges defined as indices of the points tensor, e.g., point_index, point_index2 = edge_indices[i]

    Returns
    -
    edges_point_based: [M, 2, 2] M edges defined by two points p1, p2 = edges[i]
    """
    return points[edges_index_based]  # [N_strings, 2, 2]


def angle_based(pin_angles: torch.Tensor, edges_index_based: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    -
    pin_angles: [N]             angles of the pins
    edges_index_based: [M, 2]  M edges defined as indices of the points tensor, e.g., point_index, point_index2 = edge_indices[i]

    Returns
    -
    edges_angle_based: [M, 2] M edges defined by two angles psi1, psi2 = edges[i]
    """
    return pin_angles[edges_index_based]  # [N_strings, 2]


def radon_index_based(edges_angle_based: torch.Tensor, alpha_domain: torch.Tensor, image_size: int) -> torch.Tensor:
    """
    Parameters
    -
    edges_angle_based: [N, 2]  N edges each defined by two angles psi1, psi2 = edges[i]
    alpha_domain: [M]          M possible values for radon angle alpha
    image_size: int            width/height of the square target image

    Returns
    -
    edges_radon_parameter_based: [N, 2] N edges each defined by the domain indices of their two radon parameters s_index, alpha_index = edges[i]
                                 s_index goes from 0 to image_size-1, alpha_index from 0 to M-1
    """
    radius = image_size//2  # [1]
    psi_0, psi_1 = edges_angle_based[:, 0], edges_angle_based[:, 1]  # [N], [N]
    mean_angle = (psi_0 + psi_1) / 2  # [N]
    mean_angle[(psi_1 - psi_0) > torch.pi] -= torch.pi
    alpha_continuous = mean_angle % (2*torch.pi)  # [N]
    offset = torch.abs(torch.exp(edges_angle_based * 1j).mean(dim=1)) * radius  # [N]

    s_continuous = radius * torch.ones_like(alpha_continuous)  # [N]
    mask = (alpha_continuous < torch.pi)  # [N]
    s_continuous[mask] = s_continuous[mask] + offset[mask]  # [N]
    s_continuous[~mask] = s_continuous[~mask] - offset[~mask]  # [N]
    alpha_continuous = alpha_continuous % torch.pi  # [N]

    alpha_diffs = torch.abs(alpha_continuous.unsqueeze(1) - alpha_domain.unsqueeze(0))  # [N, M]
    alpha_indices = torch.argmin(alpha_diffs, dim=1)  # [N]
    s_indices = torch.clamp(torch.round(s_continuous).to(torch.int), 0, image_size-1)  # [N]
    return torch.stack([s_indices, alpha_indices], dim=1)  # [N, 2]


def radon_parameter_based(edges_radon_index_based: torch.Tensor, alpha_domain: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    -
    edges_radon_index_based: [N, 2]  N edges each defined by the domain indices of their two radon parameters s_index, alpha_index = edges[i]
    alpha_domain: [M]                M possible values for radon angle alpha

    Returns
    -
    edges_radon_parameter_based: [N, 2]  N edges each defined by their two radon parameters s, alpha = edges[i]
    """
    alpha = alpha_domain[edges_radon_index_based[:, 1]]  # [N]
    s = edges_radon_index_based[: 0]
    return torch.stack([s, alpha], dim=1)  # [N, 2]


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
    edges: [2, 2]  edge defined by two points p_start, p_end = edges[i]
    line_profile:  function determininig the pixel intensity based on its distance to the edge
    image_size:    width/height of the square target image

    Returns
    -
    edge_image: [image_size, image_size] image of the edge. Most pixels are 0.
    """
    rows, cols = torch.meshgrid(torch.arange(image_size), torch.arange(image_size), indexing='ij')  # [image_size, image_size]
    grid = torch.stack([rows, cols], dim=-1).reshape(-1, 2).to(edge.dtype)  # [HW, 2]
    distances = distance_points_edge(grid, edge)  # [HW]
    edge_image = line_profile(distances).reshape(image_size, image_size)  # [image_size, image_size]
    return edge_image
