import torch


class StringPath:
    edges: torch.Tensor
    """[N_strings, 2]  set of possible edges to choose from where pin_index_lower, pin_index_higher = edge[i]"""
    x: torch.Tensor
    """[N_strings] binary solution vector where x[i]=1 means string i is choosen and x[i]=0 that string i is still available"""
    edge_path: list[int]
    """ordered list of edge indices as they were added to the path"""
    pin_path: list[int]
    """ordered list of pin indices as they were added to the path"""

    def __init__(self, edge_indices: torch.Tensor, start_pin_index: int) -> None:
        self.edges = edge_indices
        self.x = torch.zeros(edge_indices.shape[0])
        self.edge_path = []
        self.pin_path = [start_pin_index]

    @property
    def available_edge_indices(self) -> torch.Tensor:
        return torch.where(self.x == 0)[0]

    @property
    def available_edges(self) -> torch.Tensor:
        return self.edges[self.available_edge_indices]

    def add_edge_index(self, edge_index: int) -> None:
        self.x[edge_index] = 1
        self.edge_path.append(edge_index)
        last_pin = self.pin_path[-1]
        pin_index_1, pin_index_2 = self.edges[edge_index]
        self.pin_path.append(pin_index_2 if pin_index_1 == last_pin else pin_index_1)


def naive_greedy(string_matrix: torch.Tensor, target_image: torch.Tensor, path: StringPath) -> StringPath:
    """
    Parameters
    -
    string_matrix: [HW, N_strings]  matrix where entry (i, j) is the intensity of pixel i when string j is placed 
    target_image:  [HW]             flattened image vector where entry i is the intensity of pixel i
    path: [N_used_strings]          string path

    Returns
    path: 
    """
    Ax = string_matrix @ path.x  # [HW]
    last_pin_index = path.pin_path[-1] * torch.ones(1, 1)  # [1, 1]
    contains_last_pin_index = torch.any(last_pin_index == path.available_edges, dim=1)  # [N_available]
    edge_indices_starting_from_current_pin = path.available_edge_indices[contains_last_pin_index]  # [N_candidates]
    candidate_strings = string_matrix[:, edge_indices_starting_from_current_pin]  # [HW, N_candidates]
    b_current_reconstruction = torch.clamp(Ax.unsqueeze(1) + candidate_strings, 0, 1)  # [HW, N_candidates]
    reconstruction_error = torch.mean(b_current_reconstruction - target_image.unsqueeze(1), dim=0)**2  # [N_candidates]
    best_candidate_index = torch.argmin(reconstruction_error)  # [1]
    best_edge_index = edge_indices_starting_from_current_pin[best_candidate_index]  # [1]
    best_reconstruction_error = reconstruction_error[best_candidate_index]  # [1]
    return best_edge_index, best_reconstruction_error
