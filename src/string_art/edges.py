import torch
from string_art.line_profile import LineProfile
from string_art.pins import PinsAngleBased, PinsPointBased

type EdgeIndexBased = torch.Tensor
"""[2] edge[0]=i, edge[1]=j where i,j being pin indices in 0,...,N_pins-1"""
type EdgesIndexBased = torch.Tensor
"""[N, 2]"""

type EdgePointBased = torch.Tensor
"""[2, 2] edge[0]=[start_x, start_y], edge[1]=[end_x, end_y]"""
type EdgesPointBased = torch.Tensor
"""[N, 2, 2]"""

type EdgeAngleBased = torch.Tensor
"""[2] edge[0]=psi1, edge[1]=psi2 where psi1, psi2 being pin angles between [0, 2pi]"""
type EdgesAngleBased = torch.Tensor
"""[N, 2]"""

type EdgeRadonParameterBased = torch.Tensor
"""[2] edge[0]=s, edge[1]=alpha where s,alpha are radon parameters defining a line through two pins for s in [-1, 1] and alpha in [0, pi]"""
type EdgesRadonParameterBased = torch.Tensor
"""[N, 2]"""


def index_based(N_pins: int) -> EdgesIndexBased:
    """
    Computes all possible edges between any two pins.
    N_edges = N_pins * (N_pins - 1) / 2 = N_pins choose 2

    **Parameters**  
    N_pins: number of pins

    **Returns**
    edges_index_based: [N_edges, 2] The first entry is always the lower pin index, i.e., pin_index_lower, pin_index_higher = edge_indices[i]
    """
    pin_indices = torch.arange(N_pins)
    string_indices = torch.combinations(pin_indices, r=2)  # [N_edges, 2]
    return string_indices


def point_based(pins_point_based: PinsPointBased, edges_index_based: EdgesIndexBased) -> EdgesPointBased:
    """
    **Parameters**  
    pins_point_based: [N_pins, 2]    N_pins two-dimensional points
    edges_index_based: [N_edges, 2]  N_edges edges each defined by ids that can be interpreted as indices of the points tensor, e.g., point_index, point_index2 = edge_indices[i]

    **Returns**  
    edges_point_based: [N_edges, 2, 2] N_edges edges each defined by two points p1, p2 = edges[i]
    """
    return pins_point_based[edges_index_based]  # [N_edges, 2, 2]


def angle_based(pins_angle_based: PinsAngleBased, edges_index_based: EdgesIndexBased) -> EdgesAngleBased:
    """
    **Parameters**  
    pin_angles: [N_pins]             N_pins angles of the pins
    edges_index_based: [N_edges, 2]  N_edges edges each defined by ids that can be interpreted as indices of the points tensor, e.g., point_index, point_index2 = edge_indices[i]

    **Returns**  
    edges_angle_based: [N_edges, 2] N_edges edges each defined by two angles psi1, psi2 = edges_angle_based[i]
    """
    return pins_angle_based[edges_index_based]  # [N_edges, 2]


def angle_to_index_based(pins_angle_based: PinsAngleBased, edges_angle_based: EdgesAngleBased) -> EdgesIndexBased:
    pins_1d = pins_angle_based.squeeze()
    matches = torch.abs(edges_angle_based.unsqueeze(-1) - pins_1d.unsqueeze(0).unsqueeze(1)) < 1e-3
    edges_index_based = torch.argmax(matches.to(torch.int), dim=-1)
    return edges_index_based

def angle_to_point_based(edges_angle_based: EdgesAngleBased) -> EdgesPointBased:
    psi1 = edges_angle_based[:, 0] # [N]
    psi2 = edges_angle_based[:, 1] # [N]
    start_points = torch.stack([torch.cos(psi1), torch.sin(psi1)], dim=1) # [N, 2]
    end_points = torch.stack([torch.cos(psi2), torch.sin(psi2)], dim=1) # [N, 2]
    return torch.stack([start_points, end_points], dim=1) # [N, 2, 2]

def angle_to_radon_parameter_based(edges_angle_based: EdgesAngleBased) -> EdgesRadonParameterBased:
    """
    Parameters
    -
    edges_angle_based: [N_edges, 2]  N edges each defined by two angles psi1, psi2 = edges[i]

    Returns
    -
    edges_radon_parameter_based: [N_edges, 2]  N edges each defined by their two radon parameters s, alpha = edges[i]
    """
    def _mean_angle(psi_0: torch.Tensor, psi_1: torch.Tensor) -> torch.Tensor:
        """Computes the mean angle between psi_0 and psi_1 in the range [0, 2*pi)"""
        mean_angle = (psi_0 + psi_1) / 2  # [N]
        mean_angle[(psi_1 - psi_0) > torch.pi] -= torch.pi  # [N]
        return mean_angle % (2*torch.pi)  # [N]

    psi_0, psi_1 = edges_angle_based[:, 0], edges_angle_based[:, 1]  # [N_edges], [N_edges]
    mean_angle = _mean_angle(psi_0, psi_1)  # [N_edges]
    s = torch.abs(torch.exp(edges_angle_based * 1j).mean(dim=1))  # [N]
    s[mean_angle >= torch.pi] *= -1  # [N]
    alpha = mean_angle % torch.pi  # [N]
    return torch.stack([s, alpha], dim=1)

def radon_parameter_to_radon_index_based(edges_radon_parameter_based: EdgesRadonParameterBased, s_domain: torch.Tensor, alpha_domain: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    -
    edges_radon_parameter_based: [N_edges, 2]  N edges each defined by their two radon parameters s, alpha = edges[i]
    s_domain: [N_s]                    possible values for the radon parameter s (distance to center) in [-1, 1]
    alpha_domain: [N_alpha]            possible values for the radon parameter alpha (angle) in [0, pi)

    Returns
    -
    edges_radon_index_based: [N_edges, 2] N edges each defined by the domain indices of their two radon parameters s_index, alpha_index = edges[i]
                             s_index from 0 to N_s-1, alpha_index from 0 to N_alpha-1
    """
    def _find_nearest_domain_index(values: torch.Tensor, domain: torch.Tensor) -> torch.Tensor:
        diffs = torch.abs(values.unsqueeze(1) - domain.unsqueeze(0))  # [N_values, N_domain]
        return torch.argmin(diffs, dim=1)  # [N_values]

    s, alpha = edges_radon_parameter_based.T  # [N_edges], [N_edges]
    s_indices = _find_nearest_domain_index(s, s_domain)  # [N_edges]
    alpha_indices = _find_nearest_domain_index(alpha, alpha_domain)  # [N_edges]
    return torch.stack([s_indices, alpha_indices], dim=1)  # [N_edges, 2]

def radon_parameter_to_angle_based(edges_radon_parameter_based: EdgesRadonParameterBased) -> EdgesAngleBased:
    """
    Parameters
    -
    edges_radon_parameter_based: [N_edges, 2]  N edges each defined by their two radon parameters s, alpha = edges[i]

    Returns
    -
    edges_angle_based: [N_edges, 2]  N edges each defined by two angles psi1, psi2 = edges[i]
    """
    s = edges_radon_parameter_based[:, 0] # [N]
    alpha = edges_radon_parameter_based[:, 1] # [N]
    alpha[s<0] += torch.pi 
    e1 = s.abs() * torch.exp(alpha*1j) # [N]
    e2 = (1-s**2).sqrt() * torch.exp((alpha + torch.pi/2)*1j) # [N]
    edges_complex_point_based = torch.stack([e1 - e2, e1 + e2], dim=1) # [N, 2]
    edges_angle_based = (torch.angle(edges_complex_point_based) + 2*torch.pi) % (2*torch.pi) # [N, 2]
    edges_angle_based[edges_angle_based > 2*torch.pi - 1e-5] = 0.0 # [N, 2] 
    edges_angle_based, _ = torch.sort(edges_angle_based, dim=1) # [N, 2]
    return edges_angle_based # [N, 2]

def distance_points_edge(points: PinsPointBased, edge: EdgePointBased) -> torch.Tensor:
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


def get_image(edge: EdgePointBased, line_profile: LineProfile, image_size: int) -> torch.Tensor:
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
